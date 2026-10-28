"""Mikrofon ve hoparlor: ham 16-bit PCM, ornekleme donusumu ALSA plug'da."""

import asyncio
import logging
import math
import subprocess
import threading
import time
from array import array
from collections import deque

import sounddevice as sd

log = logging.getLogger(__name__)

MIC_RATE = 16000      # Gemini Live girisi
SPEAKER_RATE = 24000  # Gemini Live cikisi
FRAME_MS = 20
MIC_FRAME_BYTES = MIC_RATE * FRAME_MS // 1000 * 2


def set_mic_boost(db: int, card: str) -> None:
    """~/.asoundrc softvol kontrolu: -10 dB = 0, 1 dB adim; acilista sifirlandigi icin her seferinde."""
    step = max(0, min(40, db + 10))
    r = subprocess.run(["amixer", "-q", "-c", card, "sset", "Mic Boost", str(step)],
                       capture_output=True, text=True)
    if r.returncode:
        log.warning("mic kazanci ayarlanamadi: %s", r.stderr.strip())
    else:
        log.info("mic kazanci: %+d dB", db)


class HighPass:
    """Birinci derece DC engelleyici / yuksek geciren filtre.

    INMP441 buyuk bir DC ve <2 Hz kayma uretiyor (ham -26 dBFS, temizlenince -40):
    hem enerji olcumunu hem VAD'i hem de modele giden sesi bozuyor.
    """

    def __init__(self, cutoff_hz: float = 80.0):
        self._r = 1.0 - 2 * math.pi * cutoff_hz / MIC_RATE
        self._x = 0.0
        self._y = 0.0

    def __call__(self, pcm: bytes) -> bytes:
        samples = array("h", pcm)
        r, x1, y1 = self._r, self._x, self._y
        for i, x in enumerate(samples):
            y1 = x - x1 + r * y1
            x1 = x
            samples[i] = max(-32768, min(32767, int(y1)))
        self._x, self._y = x1, y1
        return samples.tobytes()


class Mic:
    """Filtrelenmis 20 ms'lik PCM karelerini asyncio kuyruguna yazar."""

    def __init__(self, device: str, highpass_hz: float = 80.0):
        self._device = device
        self._stream = None
        self._filter = HighPass(highpass_hz)
        self.frames: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)

    def start(self) -> None:
        loop = asyncio.get_running_loop()

        def push(data: bytes) -> None:
            if self.frames.full():
                self.frames.get_nowait()
            self.frames.put_nowait(data)

        def callback(indata, frames, time_info, status):
            if status:
                log.debug("mic: %s", status)
            loop.call_soon_threadsafe(push, self._filter(bytes(indata)))

        self._stream = sd.RawInputStream(
            device=self._device, samplerate=MIC_RATE, channels=1, dtype="int16",
            blocksize=MIC_FRAME_BYTES // 2, callback=callback,
        )
        self._stream.start()
        log.info("mikrofon acildi: %s %d Hz", self._device, MIC_RATE)

    def stop(self) -> None:
        if self._stream:
            self._stream.close()
            self._stream = None


class Speaker:
    """Iki kanalli karistirici: robotun sesi ve radyo. Ses duzeyi ikisine de uygulanir.

    'busy' yalnizca robotun sesine bakar (mikrofon o surede susar); radyo calarken mikrofon
    acik kalir, konusmayi radyo yankisindan ayirmak icin radio_level_db() kullanilir.
    Robot konusurken ya da duck() suresince radyo kisilir.
    """

    def __init__(self, device: str, echo_tail_ms: int, volume: int = 100,
                 radio_level: float = 0.5, duck: float = 0.15):
        self._device = device
        self._tail = echo_tail_ms / 1000
        self._buf = bytearray()
        self._radio = bytearray()
        self._lock = threading.Lock()
        self._last_sound = 0.0
        self._duck_until = 0.0
        self._radio_level, self._duck = radio_level, duck
        self._duck_k = 1.0  # anlik kisma carpani; hedefe yavasca gider (ani ses sicramasi olmasin)
        # Son 800 ms'de calinan radyo seviyesi. Hoparlor gecikmesi yuzunden yanki mikrofona
        # gec gelir: radyo kisilinca ref hemen duser ama eski yuksek yanki hala duyulur; pencere
        # bu gecikmeyi kapsamazsa yanki "konusma" sanilir, radyo kisilir, dongu olur.
        self._radio_db = deque(maxlen=40)
        self._stream = None
        self.volume = volume

    @property
    def volume(self) -> int:
        return self._volume

    @volume.setter
    def volume(self, percent: int) -> None:
        self._volume = max(0, min(100, int(percent)))
        # Kulak logaritmik duyar: %50 ~ -12 dB
        self._gain = (self._volume / 100) ** 2

    def start(self) -> None:
        def callback(outdata, frames, time_info, status):
            n = len(outdata)
            now = time.monotonic()
            with self._lock:
                voice = self._buf[:n]
                del self._buf[:n]
                radio = self._radio[:n]
                del self._radio[:n]
            if voice:
                self._last_sound = now
            if not radio:
                self._radio_db.append(-120.0)
                if self._gain == 1.0:
                    outdata[:len(voice)] = voice
                    outdata[len(voice):] = b"\x00" * (n - len(voice))
                    return
            out = array("h", bytes(n))
            v = array("h", bytes(voice))
            g = self._gain
            for i, x in enumerate(v):
                out[i] = int(x * g)
            if radio:
                ducked = voice or now < self._duck_until or now - self._last_sound < self._tail
                # 20 ms'lik bloklarda: kisma ~100 ms, geri acilma ~1 sn surer
                target = self._duck if ducked else 1.0
                step = 0.2 if target < self._duck_k else 0.02
                self._duck_k += max(-step, min(step, target - self._duck_k))
                rg = g * self._radio_level * self._duck_k
                r = array("h", bytes(radio))
                energy = 0
                for i, x in enumerate(r):
                    y = int(x * rg)
                    energy += y * y
                    out[i] = max(-32768, min(32767, out[i] + y))
                self._radio_db.append(10 * math.log10(max(energy / len(r), 1.0) / 32768 ** 2))
            outdata[:] = out.tobytes()

        self._stream = sd.RawOutputStream(
            device=self._device, samplerate=SPEAKER_RATE, channels=1, dtype="int16",
            blocksize=SPEAKER_RATE * FRAME_MS // 1000, latency="high", callback=callback,
        )
        self._stream.start()
        log.info("hoparlor acildi: %s %d Hz, ses %%%d", self._device, SPEAKER_RATE, self._volume)

    def play(self, pcm: bytes) -> None:
        with self._lock:
            self._buf.extend(pcm)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    # --- radyo kanali ---
    def feed_radio(self, pcm: bytes) -> None:
        with self._lock:
            self._radio.extend(pcm)

    def clear_radio(self) -> None:
        with self._lock:
            self._radio.clear()

    def radio_backlog(self) -> float:
        """Tamponda bekleyen radyo sesi (sn)."""
        return len(self._radio) / (SPEAKER_RATE * 2)

    def radio_level_db(self) -> float | None:
        """Son 800 ms'de hoparlore giden en yuksek radyo seviyesi; radyo yoksa None."""
        level = max(self._radio_db, default=-120.0)
        return level if level > -90 else None

    def duck(self, seconds: float) -> None:
        """Radyoyu bu kadar sure kis (kullanici konusuyor)."""
        self._duck_until = max(self._duck_until, time.monotonic() + seconds)

    @property
    def busy(self) -> bool:
        with self._lock:
            if self._buf:
                return True
        return time.monotonic() - self._last_sound < self._tail

    def stop(self) -> None:
        if self._stream:
            self._stream.close()
            self._stream = None
