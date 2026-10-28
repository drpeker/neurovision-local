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

from .actions import Actions, you
from .audio import FRAME_MS, Mic, Speaker
from .body import Body
from .camera import Camera
from .debug import DebugRecorder
from .local_ai import LanguageModel, SpeechToText, TextToSpeech
from .memory import Memory
from .router import route
from .vad import SpeechGate

log = logging.getLogger(__name__)

# Whisper avg_logprob below these: the words were probably guessed (measured on the C270 and
# a speaker test set). Commands that change something are not run on a guess; chat asks again.
GUARDED = {"move", "track", "turn_to", "remember", "forget", "volume", "motor_speed", "shutdown", "express"}
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
        self._stt = SpeechToText(local)
        self._llm = LanguageModel(local, local["system_prompt"].strip())
        self._tts = TextToSpeech(local["voice"])
        self._min_command, self._min_chat = local["min_command_logprob"], local["min_chat_logprob"]
        self.face = None
        self.motors = None
        self._sentences: asyncio.Queue[str] = asyncio.Queue()
        self._speaking = False
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

    def attach_tracker(self, tracker) -> None:
        self.actions.tracker = tracker
        self.body.tracker = tracker
        tracker.notify = self.notify

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

    def _busy(self) -> bool:
        return self._speaking or not self._sentences.empty() or self._speaker.busy \
            or bool(self.motors and self.motors.busy)

    # --- main loop ---
    async def run(self) -> None:
        speaker_task = asyncio.create_task(self._speak_worker(), name="speech")
        try:
            await self._calibrate()
            t = await self._llm.warm_up()
            log.info("system prompt cached in %.1f s", t)
            if self._rec:
                self._rec.session(Path(self._cfg["local"]["llm_model"]).name, self._cfg["local"]["system_prompt"])
            log.info("listening")
            await self._listen()
        finally:
            speaker_task.cancel()

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
            frames, event = self.gate.feed(frame)
            if event == "start":
                log.info("speech")
                if self.face:
                    self.face.poke()
            utterance.extend(frames)
            if event == "end":
                if self._rec:
                    self._rec.start(utterance)
                    self._rec.end()
                pcm, utterance = b"".join(utterance), []
                await self._handle(pcm)
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
        log.info("heard %r (%.1f s audio, %.2f s, confidence %.2f) -> %s %s",
                 text, len(pcm) / 32000, heard_s, confidence, tool, args or "")

        if tool in GUARDED and confidence < self._min_command:
            answer = "Sorry, I'm not sure I heard that right. Please say it again."
        elif tool == "chat" and confidence < self._min_chat:
            answer = "Sorry, I didn't catch that. Could you say it again?"
        else:
            answer = await self.actions.run(tool, args)
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
        words = [w for w in re.findall(r"[a-z']{4,}", text.lower()) if w not in STOPWORDS]
        notes = [you(t) for _, _, t in self._memory.recall(" ".join(words), limit=2)] if words else []
        seen = self.actions.last_seen
        if seen and time.monotonic() - seen[0] < NOTES_MAX_AGE:
            notes.append(seen[1])
        prompt = text if not notes else "Notes: " + " ".join(notes) + f"\n\nUser: {text}"
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
        log.info("LLM first sentence %.2f s, total %.2f s%s", first[0] if first else 0,
                 time.monotonic() - t0, " (cut at token limit)" if cut else "")
        if not answer:
            answer = "I'm not sure what to say."
            self.say(answer)
        return answer
