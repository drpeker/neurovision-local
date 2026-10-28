"""Internet radyosu: ffmpeg yayini 24 kHz mono PCM'e cevirir, hoparlorun radyo kanalina akar.

ffmpeg m3u8 (HLS), mp3, aac gibi her yayini acar. Yayin koparsa birkac kez yeniden baglanir.
"""

import asyncio
import logging
import shutil

from .audio import SPEAKER_RATE, Speaker
from .memory import _fold

log = logging.getLogger(__name__)

CHUNK = SPEAKER_RATE * 2 // 10  # 100 ms
BACKLOG_S = 1.0                 # hoparlor tamponunda en fazla bu kadar radyo beklesin
GENERIC = {"radyo", "radyosu", "radyoyu", "radio", "fm", "the"}


class Radio:
    def __init__(self, speaker: Speaker, stations: dict[str, str]):
        self._speaker = speaker
        self.stations = stations
        self.playing: str | None = None
        self._task: asyncio.Task | None = None
        self._proc = None

    def find(self, query: str) -> str | None:
        """'kafa radyoyu' -> 'Kafa Radyo': istasyon adinin ayirt edici kelimeleri sorguda gecmeli."""
        q = [_fold(w) for w in query.split()]
        best, score = None, 0
        for name in self.stations:
            words = [_fold(w) for w in name.split()]
            key = [w for w in words if w not in GENERIC] or words
            hits = sum(any(t.startswith(w) or w.startswith(t) for t in q if len(t) > 2) for w in key)
            if hits > score:
                best, score = name, hits
        return best

    async def play(self, query: str) -> dict:
        name = self.find(query)
        if not name:
            return {"error": f"'{query}' adında bir istasyon yok.", "istasyonlar": list(self.stations)}
        if not shutil.which("ffmpeg"):
            return {"error": "ffmpeg kurulu değil."}
        await self.stop()
        self.playing = name
        self._task = asyncio.create_task(self._run(name, self.stations[name]), name="radio")
        log.info("radyo: %s", name)
        return {"result": f"{name} açılıyor."}

    async def _run(self, name: str, url: str) -> None:
        fails = 0
        try:
            while fails < 4:
                self._proc = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                    "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
                    "-i", url, "-vn", "-ac", "1", "-ar", str(SPEAKER_RATE), "-f", "s16le", "-",
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                got = 0
                while chunk := await self._proc.stdout.read(CHUNK):
                    got += len(chunk)
                    while self._speaker.radio_backlog() > BACKLOG_S:
                        await asyncio.sleep(0.05)
                    self._speaker.feed_radio(chunk)
                err = (await self._proc.stderr.read()).decode(errors="replace").strip()
                await self._proc.wait()
                fails = 0 if got > SPEAKER_RATE * 2 * 10 else fails + 1
                log.warning("radyo akisi kesildi (%s): %s", name, err[-160:] or "-")
                await asyncio.sleep(2 * fails)
            log.warning("radyo acilamadi: %s", name)
        finally:
            self._kill()
            self._speaker.clear_radio()
            if self.playing == name:
                self.playing = None

    def _kill(self) -> None:
        if self._proc and self._proc.returncode is None:
            self._proc.kill()

    async def stop(self) -> dict:
        was = self.playing
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.playing = None
        if was:
            log.info("radyo kapatildi")
        return {"result": f"{was} kapatıldı." if was else "Radyo zaten kapalı."}

    def status(self) -> dict:
        return {"çalan": self.playing or "kapalı", "istasyonlar": list(self.stations)}
