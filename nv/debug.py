"""Teshis kaydi: modele giden son konusmalarin sesi ve oturumun sistem talimati.

Bir hata olunca tam o girdiyi test oturumunda yeniden oynatmak icin.
logs/debug/ altinda: utt-SSDDSS.wav (son N konusma), session-SSDDSS.txt (son N oturum).
"""

import logging
import time
import wave
from pathlib import Path

from .audio import MIC_RATE

log = logging.getLogger(__name__)


class DebugRecorder:
    def __init__(self, folder: Path, keep: int):
        self._dir = folder
        self._dir.mkdir(parents=True, exist_ok=True)
        self._keep = keep
        self._frames: list[bytes] | None = None

    def _prune(self, pattern: str) -> None:
        for old in sorted(self._dir.glob(pattern))[:-self._keep]:
            old.unlink(missing_ok=True)

    def session(self, model: str, instruction: str) -> None:
        name = self._dir / f"session-{time.strftime('%Y%m%d-%H%M%S')}.txt"
        name.write_text(f"model: {model}\n\n{instruction}\n", encoding="utf-8")
        self._prune("session-*.txt")

    def start(self, frames: list[bytes]) -> None:
        self._frames = list(frames)

    def add(self, frames: list[bytes]) -> None:
        if self._frames is not None:
            self._frames.extend(frames)

    def end(self) -> None:
        if not self._frames:
            self._frames = None
            return
        name = self._dir / f"utt-{time.strftime('%Y%m%d-%H%M%S')}.wav"
        with wave.open(str(name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(MIC_RATE)
            w.writeframes(b"".join(self._frames))
        log.info("konusma kaydedildi: %s (%.1f sn)", name.name, len(self._frames) * 0.02)
        self._frames = None
        self._prune("utt-*.wav")
