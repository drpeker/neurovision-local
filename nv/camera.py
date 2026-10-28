"""Pi kamera: rpicam-vid MJPEG akisindan son JPEG kareyi ve kare meta verisini tutar.

Meta veri (libcamera'nin lux tahmini, pozlama, kazanc) bir FIFO uzerinden okunur;
ek kutuphane ya da buyuyen dosya yok.
"""

import asyncio
import logging
import os
import shutil
import time
from collections import deque
from pathlib import Path

log = logging.getLogger(__name__)

SOI, EOI = b"\xff\xd8", b"\xff\xd9"
META_KEYS = {"Lux": "lux", "ExposureTime": "exposure_us", "AnalogueGain": "gain",
             "ColourTemperature": "colour_k"}


class Camera:
    def __init__(self, width: int, height: int, fps: int, rotation: int = 0, hflip: bool = False):
        runtime = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))
        self._fifo = runtime / f"nv-camera-meta-{os.getpid()}"
        # rpicam yalnizca 0/180 dondurur; 90/270 (yan takili kamera) kare istendiginde PIL ile
        self._soft_rotation = rotation if rotation in (90, 270) else 0
        self._rotated: tuple[float, bytes] | None = None
        self._args = [
            "-t", "0", "-n", "--codec", "mjpeg", "--quality", "70",
            "--width", str(width), "--height", str(height),
            "--framerate", str(fps), "--rotation", "180" if rotation == 180 else "0", "--flush",
            "--metadata", str(self._fifo), "--metadata-format", "txt", "-o", "-",
        ]
        if hflip:
            self._args.insert(-2, "--hflip")
        self._proc = None
        self._task = None
        self._meta_task = None
        self.latest: bytes | None = None
        self.latest_at = 0.0
        self.meta: dict[str, float] = {}
        self.meta_at = 0.0

    async def start(self) -> None:
        exe = shutil.which("rpicam-vid")
        if not exe:
            log.warning("rpicam-vid bulunamadi, kamera kapali")
            return
        self._fifo.unlink(missing_ok=True)
        os.mkfifo(self._fifo)
        self._meta_task = asyncio.create_task(self._read_meta(), name="camera-meta")
        self._task = asyncio.create_task(self._supervise(exe), name="camera")

    async def _read_meta(self) -> None:
        # O_RDWR: yazan taraf (rpicam-vid) yeniden baslasa da okuma EOF gormez
        fd = os.open(self._fifo, os.O_RDWR | os.O_NONBLOCK)
        reader = asyncio.StreamReader()
        loop = asyncio.get_running_loop()
        await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader),
                                     os.fdopen(fd, "rb", buffering=0))
        while True:
            key, _, value = (await reader.readline()).decode(errors="replace").strip().partition("=")
            if key in META_KEYS:
                try:
                    self.meta[META_KEYS[key]] = float(value)
                    self.meta_at = time.monotonic()
                except ValueError:
                    pass

    async def _supervise(self, exe: str) -> None:
        # rpicam-vid kapanirsa (kamera mesgul, kablo, surucu) birkac saniye sonra yeniden ac
        delay = 2
        while True:
            self._proc = await asyncio.create_subprocess_exec(
                exe, *self._args,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            log.info("kamera acildi: %s", " ".join(self._args[7:13]))
            err: deque[str] = deque(maxlen=3)
            drain = asyncio.create_task(self._drain(err))
            got_frame = await self._read()
            await self._proc.wait()
            await drain
            log.warning("kamera akisi kapandi (kod %s): %s", self._proc.returncode,
                        " | ".join(err) or "-")
            delay = 2 if got_frame else min(delay * 2, 60)
            await asyncio.sleep(delay)

    async def _drain(self, keep: deque) -> None:
        # stderr okunmazsa boru dolar ve rpicam-vid kilitlenir; son satirlar hata nedeni icin
        async for line in self._proc.stderr:
            line = line.decode(errors="replace").strip()
            if line:
                keep.append(line[-160:])

    async def _read(self) -> bool:
        buf = bytearray()
        got_frame = False
        while True:
            chunk = await self._proc.stdout.read(65536)
            if not chunk:
                return got_frame
            buf.extend(chunk)
            # Tampondaki en son tam JPEG'i al, gerisini at
            end = buf.rfind(EOI)
            if end < 0:
                continue
            start = buf.rfind(SOI, 0, end)
            if start >= 0:
                self.latest = bytes(buf[start:end + 2])
                self.latest_at = time.monotonic()
                got_frame = True
            del buf[:end + 2]

    def frame(self, max_age: float = 3.0) -> bytes | None:
        if not self.latest or time.monotonic() - self.latest_at > max_age:
            return None
        if not self._soft_rotation:
            return self.latest
        # Ayni kare icin bir kez dondur
        if self._rotated and self._rotated[0] == self.latest_at:
            return self._rotated[1]
        import io

        from PIL import Image
        img = Image.open(io.BytesIO(self.latest))
        # rotation saat yonunde derece; PIL transpose saat yonunun tersine
        op = Image.Transpose.ROTATE_270 if self._soft_rotation == 90 else Image.Transpose.ROTATE_90
        out = io.BytesIO()
        img.transpose(op).save(out, format="JPEG", quality=80)
        self._rotated = (self.latest_at, out.getvalue())
        return self._rotated[1]

    async def stop(self) -> None:
        for task in (self._task, self._meta_task):
            if task:
                task.cancel()
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            await self._proc.wait()
        self._fifo.unlink(missing_ok=True)
