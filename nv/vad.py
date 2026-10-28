"""Yerel dinleme filtresi: ortam sesi modele gitmez, yalnizca onaylanan konusma gider.

Bir kare "konusma" sayilmasi icin iki sart: webrtcvad konusma demeli VE enerjisi
acilista olculen ortam tabaninin margin_db ustunde olmali. Ses once aday olarak
bekletilir; min_speech_ms kadar konusma birikmeden hicbir sey gonderilmez, birikmezse
aday atilir (oksuruk, tik, kapi, kisa TV sesi).

Radyo calarken ucuncu sart: mikrofon, hoparlore giden radyonun beklenen yankisinin
echo_margin_db ustunde olmali. Yanki orani (mikrofon - radyo, dB) konusma yokken ogrenilir.
"""

import math
from array import array
from collections import deque

import webrtcvad

from .audio import FRAME_MS, MIC_RATE

# Adayin onaylanmasi icin taninan sure: min_speech_ms'in bu kati icinde birikmeli
CONFIRM_WINDOW = 2.5
# Sessiz anlarda ortam tabaninin yeni seviyeye kayma hizi (kare basina)
FLOOR_ADAPT = 0.01
# Yanki orani ogrenme hizi: yukari hizli, asagi yavas -> yankinin tepelerini izler.
# Esigi gecen kareler ogrenmeye katilmaz, konusma orani sisirmez.
ECHO_UP, ECHO_DOWN = 0.1, 0.02


def frame_db(frame: bytes) -> float:
    samples = array("h", frame)
    rms = math.sqrt(sum(s * s for s in samples) / len(samples)) if samples else 0.0
    return 20 * math.log10(max(rms, 1.0) / 32768)


class SpeechGate:
    """feed(kare) -> (gonderilecek_kareler, olay); olay "start", "end" ya da None."""

    def __init__(self, aggressiveness: int, preroll_ms: int, hangover_ms: int,
                 min_speech_ms: int = 300, margin_db: float = 10.0, echo_margin_db: float = 10.0):
        self._vad = webrtcvad.Vad(aggressiveness)
        self._preroll = deque(maxlen=max(1, preroll_ms // FRAME_MS))
        self._hangover = max(1, hangover_ms // FRAME_MS)
        self._need = max(1, min_speech_ms // FRAME_MS)
        self._window = int(self._need * CONFIRM_WINDOW)
        self._margin = margin_db
        self._echo_margin = echo_margin_db
        self.floor_db = -60.0
        self.echo_ratio_db: float | None = None  # mikrofon - radyo; radyo hic calmadiysa None
        self.reset()

    def calibrate(self, frames: list[bytes]) -> float:
        """Ortam tabani: kare enerjilerinin medyani (arada gecen kisa bir ses sapitmasin)."""
        levels = sorted(frame_db(f) for f in frames)
        self.floor_db = levels[len(levels) // 2]
        return self.floor_db

    def reset(self) -> None:
        self._preroll.clear()
        self._candidate: list[bytes] = []
        self._voiced = 0
        self._silent = 0
        self.active = False

    def _over_echo(self, db: float, ref_db: float | None) -> bool:
        """Radyo caliyorsa: mikrofon radyonun yankisindan belirgin yuksek mi?"""
        if ref_db is None:
            return True
        ratio = db - ref_db
        if self.echo_ratio_db is None:
            self.echo_ratio_db = ratio
        over = ratio - self.echo_ratio_db >= self._echo_margin
        if not over and not self.active:
            k = ECHO_UP if ratio > self.echo_ratio_db else ECHO_DOWN
            self.echo_ratio_db += (ratio - self.echo_ratio_db) * k
        return over

    def _is_speech(self, frame: bytes, ref_db: float | None) -> tuple[bool, float]:
        db = frame_db(frame)
        # webrtcvad kendi icinde durum tutuyor: her karede cagrilmali, sonuc sonra birlestirilir
        voiced = self._vad.is_speech(frame, MIC_RATE)
        over_echo = self._over_echo(db, ref_db)
        return voiced and over_echo and db >= self.floor_db + self._margin, db

    def feed(self, frame: bytes, ref_db: float | None = None) -> tuple[list[bytes], str | None]:
        """ref_db: hoparlorden su an calan radyonun seviyesi (yoksa None)."""
        speech, db = self._is_speech(frame, ref_db)

        if self.active:
            self._silent = 0 if speech else self._silent + 1
            if self._silent >= self._hangover:
                self.reset()
                return [frame], "end"
            return [frame], None

        if self._candidate:
            self._candidate.append(frame)
            self._voiced += speech
            # Radyo calarken onay icin iki kat konusma gereksin (muzikteki vokaller kisa patlar)
            if self._voiced >= self._need * (2 if ref_db is not None else 1):
                self.active = True
                frames, self._candidate = self._candidate, []
                return frames, "start"
            if len(self._candidate) >= self._window:
                # Onaylanmadi: kisa gurultu, at; son kareler on tampon olarak kalsin
                self._preroll.extend(self._candidate[-self._preroll.maxlen:])
                self._candidate, self._voiced = [], 0
            return [], None

        if speech:
            self._candidate = list(self._preroll) + [frame]
            self._voiced = 1
            self._preroll.clear()
        else:
            self._preroll.append(frame)
            # Ortam degisirse (klima, fan) taban yavasca yeni seviyeye kayar; radyo tabani bozmasin
            if ref_db is None and db < self.floor_db + self._margin:
                self.floor_db += (db - self.floor_db) * FLOOR_ADAPT
        return [], None
