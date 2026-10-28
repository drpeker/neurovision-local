"""Gemini Live oturumu: ses gonder/al, araclar, kesintisiz yeniden baglanma."""

import asyncio
import logging
import time
from pathlib import Path

from google import genai
from google.genai import types

from .audio import FRAME_MS, MIC_RATE, Mic, Speaker
from .body import Body
from .camera import Camera
from .debug import DebugRecorder
from .memory import Memory
from .summarizer import consolidate
from .tools import Tools
from .vad import SpeechGate

log = logging.getLogger(__name__)

AUDIO_MIME = f"audio/pcm;rate={MIC_RATE}"


class LiveBrain:
    def __init__(self, cfg: dict, api_key: str, mic: Mic, speaker: Speaker,
                 camera: Camera | None, memory: Memory):
        self._cfg = cfg
        self._client = genai.Client(api_key=api_key)
        self._mic = mic
        self._speaker = speaker
        self._camera = camera
        self._memory = memory
        v = cfg["vad"]
        self.gate = SpeechGate(v["aggressiveness"], v["preroll_ms"], v["hangover_ms"],
                               v["min_speech_ms"], v["margin_db"], cfg["radio"]["echo_margin_db"])
        self.body = Body(cfg, camera, self.gate, memory)
        self.body.speaker = speaker
        self._tools = Tools(camera, memory, self.body, speaker)
        self._tools.heard = lambda: "".join(self._heard)
        self.face = None  # attach_face ile (nv.display.Face)
        self.motors = None  # attach_motors ile (nv.motors.Motors)
        self._frame_on_speech = cfg["camera"]["enabled"] and cfg["camera"]["frame_on_speech"]
        d = cfg["debug"]
        self._rec = DebugRecorder(Path(d["dir"]), d["keep"]) if d["enabled"] else None
        # Oturum devami yalnizca calisirken (GoAway, ag kopmasi) kullanilir; yeniden baslayinca
        # temiz oturum acilir, sureklilik hafiza baglamiyla saglanir
        self._handle: str | None = None
        self._last_activity = time.monotonic()
        self._heard: list[str] = []
        self._said: list[str] = []
        self._renew: asyncio.Event | None = None  # acik oturumu yenilemek icin (_session doldurur)
        self._live = None  # acik oturum (notify icin)

    def attach_face(self, face) -> None:
        self.face = face
        self._tools.face = face

    def attach_motors(self, motors) -> None:
        self.motors = motors
        self._tools.motors = motors
        self.body.motors = motors

    def attach_tracker(self, tracker) -> None:
        self._tools.tracker = tracker
        self.body.tracker = tracker
        tracker.notify = self.notify

    async def notify(self, text: str) -> None:
        """Arka plandaki bir isten (takip) modele kisa olay bildirimi; model sesli soyler."""
        if self._live:
            log.info("bildirim: %s", text)
            await self._live.send_realtime_input(text=f"[Sistem olayı] {text}")

    def attach_shutdown(self, fn) -> None:
        self._tools.on_shutdown = fn

    def attach_radio(self, radio) -> None:
        self._tools.radio = radio
        self.body.radio = radio

    def _instruction(self) -> str:
        m = self._cfg["memory"]
        text = self._cfg["gemini"]["system_instruction"].strip()
        context = self._memory.context(m["facts_in_prompt"], m["episodes_in_prompt"])
        return f"{text}\n\n{context}" if context else text

    def _config(self) -> types.LiveConnectConfig:
        g = self._cfg["gemini"]
        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=self._instruction(),
            speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=g["voice"]))),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    prefix_padding_ms=300, silence_duration_ms=700)),
            # Sikistirma: baglam trigger_tokens'i gecince en eskisi atilip target_tokens'e iner;
            # her tur tum baglami yeniden isledigi icin tur basina token bununla sinirli kalir
            context_window_compression=types.ContextWindowCompressionConfig(
                trigger_tokens=g["context_trigger_tokens"],
                sliding_window=types.SlidingWindow(target_tokens=g["context_target_tokens"])),
            # resumption: baglanti kopunca (GoAway, ag) ayni oturuma donulur
            session_resumption=types.SessionResumptionConfig(handle=self._handle),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
            tools=self._tool_list(),
        )

    def _tool_list(self) -> list[types.Tool]:
        tools = [types.Tool(function_declarations=self._tools.declarations())]
        if self._cfg["gemini"].get("google_search"):
            # Sunucuda calisir, arac cagrisi bize gelmez; ucretsiz katmanda kotasi yok
            tools.append(types.Tool(google_search=types.GoogleSearch()))
        return tools

    async def run(self) -> None:
        await self._calibrate()
        consolidator = asyncio.create_task(self._consolidate_when_idle(), name="memory")
        try:
            await self._connect_forever()
        finally:
            consolidator.cancel()

    async def _calibrate(self) -> None:
        """Acilista ortami dinle; konusma bu taban seviyenin belirgin ustunde olmali."""
        n = max(1, self._cfg["vad"]["calibrate_ms"] // FRAME_MS)
        # Mikrofonun ilk yarim saniyesi acilis dalgalanmasi: olcume katma
        for _ in range(500 // FRAME_MS):
            await self._mic.frames.get()
        frames = [await self._mic.frames.get() for _ in range(n)]
        floor_db = self.gate.calibrate(frames)
        log.info("ortam gurultusu %.1f dBFS; konusma esigi %.1f dBFS",
                 floor_db, floor_db + self._cfg["vad"]["margin_db"])

    async def _consolidate_when_idle(self) -> None:
        """Orta/uzun hafiza: sohbet sustugunda dokumu ozetle, kalici bilgileri cikar.

        Ozetten sonra oturum yenilenir: yeni oturum yalnizca talimat + ozetler + bilgilerle
        baslar, eski sohbet her turda yeniden islenip token harcamaz.
        """
        m = self._cfg["memory"]
        while True:
            await asyncio.sleep(30)
            if time.monotonic() - self._last_activity < m["idle_s"]:
                continue
            try:
                summarized = await consolidate(self._memory, self._client, m["summary_model"])
            except Exception as e:
                log.warning("hafiza ozetleme hatasi: %s", e)
                continue
            busy = self.gate.active or self._speaker.busy or (self.motors and self.motors.busy)
            if summarized and self._renew and not busy:
                log.info("sessizlik: yeni oturum aciliyor (sohbet ozetlendi)")
                self._handle = None
                self._renew.set()

    async def _connect_forever(self) -> None:
        delay = 1
        while True:
            try:
                await self._session()
                delay = 1
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("oturum hatasi: %s; %d sn sonra yeniden baglaniyor", e, delay)
                if self._handle and delay >= 2:
                    # Iki kez ust uste basarisizsa belirtec gecersiz olabilir: yeni oturum ac
                    log.info("oturum belirteci birakildi, yeni oturum acilacak")
                    self._handle = None
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)

    async def _session(self) -> None:
        model = self._cfg["gemini"]["model"]
        config = self._config()
        async with self._client.aio.live.connect(model=model, config=config) as s:
            log.info("Gemini Live baglandi (%s, %s)", model,
                     "devam" if self._handle else "yeni oturum")
            if self._rec and not self._handle:
                self._rec.session(model, config.system_instruction)
            self.body.connected_since = time.monotonic()
            self._live = s
            if self._handle:
                # Baglanti konusmanin ortasinda koptuysa sunucu o konusmanin bitmesini
                # bekleyip takili kaliyor; akisin bittigini hemen bildir
                await s.send_realtime_input(audio_stream_end=True)
            renew = self._renew = asyncio.Event()
            tasks = [asyncio.create_task(self._send(s)), asyncio.create_task(self._recv(s, renew)),
                     asyncio.create_task(renew.wait())]
            try:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    if t.exception():
                        raise t.exception()
            finally:
                self.body.connected_since = None
                self._live = None
                if self.motors:
                    # Beyinle baglanti koptu: robot kontrolsuz hareket etmesin
                    self.motors.halt()
                for t in tasks:
                    t.cancel()
        log.info("baglanti yenileniyor")

    async def _send(self, s) -> None:
        self.gate.reset()
        while True:
            frame = await self._mic.frames.get()
            if self._speaker.busy or (self.motors and self.motors.busy):
                # Yari cift yonlu: robot konusurken ya da motorlar donerken mikrofon susar;
                # kendi sesi / motor sesi modele gitmez, token harcanmaz
                if self.gate.active:
                    await s.send_realtime_input(audio_stream_end=True)
                    if self._rec:
                        self._rec.end()
                self.gate.reset()
                continue

            # Radyo caliyorsa seviyesi: radyonun yankisi konusma sayilmasin
            frames, event = self.gate.feed(frame, self._speaker.radio_level_db())
            if self.gate.active or event:
                self._speaker.duck(3)  # kullanici konusurken radyo kisik
            if event == "start":
                log.info("konusma onaylandi, modele gonderiliyor")
                if self._rec:
                    self._rec.start([])
                jpeg = self._camera.frame() if self._frame_on_speech and self._camera else None
                if jpeg:
                    await s.send_realtime_input(video=types.Blob(data=jpeg, mime_type="image/jpeg"))
            for f in frames:
                await s.send_realtime_input(audio=types.Blob(data=f, mime_type=AUDIO_MIME))
            if self._rec and frames:
                self._rec.add(frames)
            if event == "end":
                await s.send_realtime_input(audio_stream_end=True)
                if self._rec:
                    self._rec.end()
                if self.face:
                    self.face.thinking = True  # cevap sesi gelene kadar "dusunuyor"

    async def _recv(self, s, renew: asyncio.Event) -> None:
        while True:
            async for msg in s.receive():
                await self._on_message(s, msg, renew)

    async def _on_message(self, s, msg: types.LiveServerMessage, renew: asyncio.Event) -> None:
        upd = msg.session_resumption_update
        if upd and upd.resumable and upd.new_handle:
            self._handle = upd.new_handle

        if msg.usage_metadata:
            self._memory.add_usage("live", msg.usage_metadata)

        if msg.go_away:
            log.info("sunucu baglantiyi kapatacak (kalan %s), yenileniyor", msg.go_away.time_left)
            renew.set()

        if msg.tool_call:
            if self.face:
                self.face.thinking = True  # arac calisirken bekleme suresi yeniden baslasin
            replies = [await self._tools.handle(fc) for fc in msg.tool_call.function_calls or []]
            await s.send_tool_response(function_responses=replies)

        sc = msg.server_content
        if not sc:
            return
        if sc.interrupted:
            self._speaker.clear()
        if sc.model_turn:
            for part in sc.model_turn.parts or []:
                if part.inline_data and part.inline_data.data:
                    self._speaker.play(part.inline_data.data)
                    if self.face:
                        self.face.thinking = False
        gm = sc.grounding_metadata
        if gm and gm.web_search_queries:
            log.info("web aramasi: %s", ", ".join(gm.web_search_queries))
        if sc.input_transcription and sc.input_transcription.text:
            self._heard.append(sc.input_transcription.text)
            if self.face:
                self.face.caption("".join(self._heard), "user")
        if sc.output_transcription and sc.output_transcription.text:
            self._said.append(sc.output_transcription.text)
            if self.face:
                self.face.caption("".join(self._said), "robot")
        if sc.turn_complete:
            if self.face:
                self.face.thinking = False
            heard, said = "".join(self._heard).strip(), "".join(self._said).strip()
            if heard:
                log.info("kullanici: %s", heard)
                self._memory.add_turn("user", heard)
            if said:
                log.info("robot: %s", said)
                self._memory.add_turn("robot", said)
            self._heard.clear()
            self._said.clear()
            self._last_activity = time.monotonic()
