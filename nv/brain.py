"""Local brain: replaces NeuroVision's Gemini Live session with models running on the Pi.

    microphone -> SpeechGate (webrtcvad) -> Whisper -> router -+-> actions (instant)  -+-> Piper -> speaker
                                                              +-> Qwen chat (stream) -+

Same attach_* hooks as the old LiveBrain, so robot.py, the face, motors and tracker are wired
up the same way. Half duplex as before: while the robot speaks or its motors run, the
microphone is ignored.
"""

import asyncio
import logging
import re
import time
from pathlib import Path

from .actions import Actions
from .audio import FRAME_MS, Mic, Speaker
from .body import Body
from .camera import Camera
from .debug import DebugRecorder
from .local_ai import LanguageModel, SpeechToText, TextToSpeech, latin
from .memory import Memory
from .router import route
from .vad import SpeechGate, frame_db

log = logging.getLogger(__name__)

# Whisper avg_logprob below these: the words were probably guessed (measured on the C270 and
# a speaker test set). Commands that change something are not run on a guess; chat asks again.
GUARDED = {"move", "track", "turn_to", "remember", "forget", "volume", "mic", "motor_speed", "shutdown", "express"}
# Not undoable: these need the stricter min_risky_logprob
RISKY = {"shutdown", "forget"}
NOTES_MAX_AGE = 120  # s; what the robot last saw is given to the LLM for this long
# Memory search (nv/memory.py) was written for Turkish: English filler words matched every
# fact ("the" -> any fact with "the"), so a joke request got a note about the user's keys
STOPWORDS = set("""about after again also always another any around because been before being both
could does doing done down each even every from have having here into just know like make many
more most much never only other over please really said same should some still such tell than
that their them then there these they thing think this those through very want were what when
where which while will with would your yours you're what's that's it's don't can't""".split())


class LocalBrain:
    def __init__(self, cfg: dict, mic: Mic, speaker: Speaker, camera: Camera | None, memory: Memory):
        self._cfg = cfg
        self._mic = mic
        self._speaker = speaker
        self._memory = memory
        local = cfg["local"]
        v = cfg["vad"]
        self.gate = SpeechGate(v["aggressiveness"], v["preroll_ms"], v["hangover_ms"],
                               v["min_speech_ms"], v["margin_db"])
        self.body = Body(cfg, camera, self.gate, memory)
        self.body.speaker = speaker
        self.actions = Actions(cfg, camera, memory, self.body, speaker)
        self.actions.wait_spoken = self._wait_spoken
        self.actions.gate = self.gate
        self._stt = SpeechToText(local)
        self._llm = LanguageModel(local, local["system_prompt"].strip())
        self._tts = TextToSpeech(local["voice"])
        self._min_command, self._min_chat = local["min_command_logprob"], local["min_chat_logprob"]
        self._min_risky = local["min_risky_logprob"]
        self._llm_ready = False
        self.face = None
        self.motors = None
        self._turner = None  # nv.imu.AngleTurner when the MPU6050 is there
        self._sentences: asyncio.Queue[str] = asyncio.Queue()
        self._speaking = False
        self._routing = False  # speech heard, not yet understood
        # Motor noise in the mic while moving (dBFS, learnt while moving): speech must be above it
        self._motor_db = cfg["vad"]["motor_noise_db"]
        d = cfg["debug"]
        self._rec = DebugRecorder(Path(d["dir"]), d["keep"]) if d["enabled"] else None

    # --- wiring (same hooks as NeuroVision's LiveBrain) ---
    def attach_face(self, face) -> None:
        self.face = face
        self.actions.face = face

    def attach_motors(self, motors) -> None:
        self.motors = motors
        self.actions.motors = motors
        self.body.motors = motors

    def attach_turner(self, turner) -> None:
        self.actions.turner = turner
        self._turner = turner

    def attach_tracker(self, tracker) -> None:
        tracker.turner = self._turner
        self.actions.tracker = tracker
        self.body.tracker = tracker
        tracker.notify = self.notify
        tracker.hold = self.hearing

    def attach_detector(self, detector) -> None:
        self.actions.detector = detector

    def attach_shutdown(self, fn) -> None:
        self.actions.on_shutdown = fn

    async def notify(self, text: str) -> None:
        """Background events (tracking found/lost the target) are spoken as they are."""
        log.info("event: %s", text)
        self.say(text)

    # --- speaking ---
    def say(self, sentence: str) -> None:
        if sentence.strip():
            self._sentences.put_nowait(sentence.strip())

    async def _speak_worker(self) -> None:
        # Piper prepares the next sentence while the speaker still plays the previous one
        while True:
            sentence = await self._sentences.get()
            self._speaking = True
            try:
                self._speaker.play(await self._tts.synth(sentence))
                if self.face:
                    self.face.thinking = False
            except Exception as e:
                log.warning("speech failed: %s", e)
            finally:
                self._sentences.task_done()
                self._speaking = not self._sentences.empty()

    async def _wait_spoken(self) -> None:
        await self._sentences.join()
        while self._speaker.busy:
            await asyncio.sleep(0.05)

    def hearing(self) -> bool:
        """The user is speaking, or what they said is not understood yet: following waits with turns."""
        return self.gate.active or self._routing

    def _following(self) -> bool:
        tracker = self.actions.tracker
        return bool(tracker and tracker.active)

    def _busy(self) -> bool:
        """Half duplex for the robot's own voice only. The mic stays open while the motors run
        (muting it then made "stop" go unheard during following); motor noise is handled by
        _min_db instead."""
        return self._speaking or not self._sentences.empty() or self._speaker.busy

    def _min_db(self, frame: bytes) -> float | None:
        """While the motors run, speech must be clearly louder than their noise."""
        if not (self.motors and self.motors.moving):
            return None
        if not self.gate.active:
            # Lower envelope of the noise: slow up, faster down, so speech over the motors does
            # not raise it (rising with every loud frame it swallowed a "stop" said while turning)
            db = frame_db(frame)
            self._motor_db += (db - self._motor_db) * (0.005 if db > self._motor_db else 0.05)
        return self._motor_db + self._cfg["vad"]["motor_margin_db"]

    # --- main loop ---
    async def run(self, llm_loading) -> None:
        """llm_loading: awaitable that finishes when the Qwen server is up (it loads meanwhile)."""
        speaker_task = asyncio.create_task(self._speak_worker(), name="speech")
        llm_task = asyncio.create_task(self._start_llm(llm_loading), name="llm")
        try:
            # Not listening and the logo on the screen until everything is loaded: "I'm ready" came
            # ~30 s before chat worked
            try:
                await llm_task
            except Exception as e:
                log.warning("language model did not start, commands only: %s", e)
            await self._calibrate()  # room noise, now that loading is over
            if self._rec:
                self._rec.session(Path(self._cfg["local"]["llm_model"]).name, self._cfg["local"]["system_prompt"])
            # The face shows "offline" (flat gray eyes) until this is set; Gemini Live set it on
            # connect, here it means everything is loaded and the robot is listening
            self.body.connected_since = time.monotonic()
            if self.face:
                self.face.ready()
            self.say("I'm ready.")  # the mic is off while the robot speaks
            await self._wait_spoken()
            while not self._mic.frames.empty():
                self._mic.frames.get_nowait()  # nothing heard during loading counts
            log.info("fully ready, listening")
            await self._listen()
        finally:
            self.body.connected_since = None
            llm_task.cancel()
            speaker_task.cancel()

    async def _start_llm(self, loading) -> None:
        t0 = time.monotonic()
        await loading
        t = await self._llm.warm_up()
        self._llm_ready = True
        log.info("language model ready %.1f s after speech recognition (system prompt cached in %.1f s)",
                 time.monotonic() - t0, t)

    async def _calibrate(self) -> None:
        """Measure room noise at start; speech must be clearly above it (same as NeuroVision)."""
        n = max(1, self._cfg["vad"]["calibrate_ms"] // FRAME_MS)
        for _ in range(500 // FRAME_MS):  # the first half second of the microphone is unsettled
            await self._mic.frames.get()
        frames = [await self._mic.frames.get() for _ in range(n)]
        floor_db = self.gate.calibrate(frames)
        log.info("room noise %.1f dBFS; speech threshold %.1f dBFS", floor_db, floor_db + self._cfg["vad"]["margin_db"])

    async def _listen(self) -> None:
        utterance: list[bytes] = []
        while True:
            frame = await self._mic.frames.get()
            if self._busy():
                # Half duplex: the robot's own voice and motor noise are not heard as speech
                self.gate.reset()
                utterance.clear()
                continue
            frames, event = self.gate.feed(frame, min_db=self._min_db(frame))
            if event == "start":
                log.info("speech")
                if self.face:
                    self.face.poke()
            utterance.extend(frames)
            if self.gate.active and self._following() and len(utterance) * FRAME_MS / 1000 > self._cfg["vad"]["max_command_s"]:
                # While following, "stop stop stop..." with no pause kept the sentence open for 12 s:
                # cut it, a command is short
                self.gate.reset()
                event = "end"
            if event == "end":
                if self._rec:
                    self._rec.start(utterance)
                    self._rec.end()
                pcm, utterance = b"".join(utterance), []
                self._routing = True
                try:
                    await self._handle(pcm)
                finally:
                    self._routing = False
                # Drop audio that arrived while thinking; it was not meant as a new request
                while not self._mic.frames.empty():
                    self._mic.frames.get_nowait()
                self.gate.reset()

    async def _handle(self, pcm: bytes) -> None:
        if self.face:
            self.face.thinking = True
        t0 = time.monotonic()
        try:
            text, confidence = await self._stt.transcribe(pcm)
        except Exception as e:
            log.warning("speech recognition failed: %s", e)
            return
        heard_s = time.monotonic() - t0
        if not text or text.strip(" .") in ("", "[BLANK_AUDIO]") or text.startswith(("[", "(")):
            log.info("nothing recognised (%.2f s)", heard_s)
            if self.face:
                self.face.thinking = False
            return
        if self.face:
            self.face.caption(text, "user")
        self._memory.add_turn("user", text)

        tool, args = route(text)
        if confidence <= -9.0 and route(latin(text))[0] == "stop":
            tool, args = "stop", {}  # "Стоп, стоп": a "stop" written in another script
        self._routing = False  # understood: following may turn again (a "stop" has already been routed)
        log.info("heard %r (%.1f s audio, %.2f s, confidence %.2f) -> %s %s",
                 text, len(pcm) / 32000, heard_s, confidence, tool, args or "")

        if tool == "stop":
            answer = await self.actions.run(tool, args)  # never doubted: stopping is always safe
        elif tool in RISKY and confidence < self._min_risky or tool in GUARDED and confidence < self._min_command:
            answer = "Sorry, I'm not sure I heard that right. Please say it again."
        elif tool == "chat" and confidence < self._min_chat:
            answer = "Sorry, I didn't catch that. Could you say it again?"
        else:
            answer = await self.actions.run(tool, args)
            if answer is not None:
                self._llm.remember(text, answer)
        if answer is not None:
            self.say(answer)
        else:
            answer = await self._chat(text)
        log.info("answer (%.1f s after speech): %s", time.monotonic() - t0, answer)
        if self.face:
            self.face.caption(answer, "robot")
        self._memory.add_turn("robot", answer)
        await self._wait_spoken()
        log.info("turn done in %.1f s, listening", time.monotonic() - t0)

    async def _chat(self, text: str) -> str:
        """Qwen answers; a few notes (matching memories, what it just saw) go with the question."""
        if not self._llm_ready:
            answer = "I'm still waking up. I can already move, look and follow, but I can't chat yet."
            self.say(answer)
            return answer
        words = [w for w in re.findall(r"[a-z']{4,}", text.lower()) if w not in STOPWORDS]
        # Facts stay in the third person ("The user's name is Mustafa."): turned into "Your name
        # is Mustafa." the model read "your" as itself and answered "My name is Mustafa."
        facts = [t for _, _, t in self._memory.recall(" ".join(words), limit=2)] if words else []
        notes = ["About the user: " + " ".join(facts)] if facts else []
        seen = self.actions.last_seen
        if seen and time.monotonic() - seen[0] < NOTES_MAX_AGE:
            notes.append("What you saw recently: " + seen[1])
        prompt = text if not notes else "\n".join(notes) + f"\n\nUser: {text}"
        t0 = time.monotonic()
        first: list[float] = []

        def on_sentence(sentence: str) -> None:
            if not first:
                first.append(time.monotonic() - t0)
            self.say(sentence)

        try:
            answer, cut = await self._llm.chat(prompt, on_sentence)
        except Exception as e:
            log.warning("LLM failed: %s", e)
            answer = "Sorry, my thinking stopped for a moment."
            self.say(answer)
            return answer
        tm = self._llm.timings
        log.info("LLM first sentence %.2f s, total %.2f s%s; prompt %s new tokens in %.2f s", first[0] if first else 0,
                 time.monotonic() - t0, " (cut at token limit)" if cut else "", tm.get("prompt_n", "?"),
                 tm.get("prompt_ms", 0) / 1000)
        if not answer:
            answer = "I'm not sure what to say."
            self.say(answer)
        return answer
