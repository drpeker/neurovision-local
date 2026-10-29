"""Yerel nesne tanima: YOLO11n (320, ONNX, COCO 80 sinif).

Model yalnizca takip sirasinda calisir; ilk kullanimda yuklenir. Cikti: sinif, guven ve
kutunun goruntuye gore oranlari (0-1). Goruntu sunmesin diye kare, kenar boslugu
eklenerek (letterbox) kare haline getirilir.
"""

import asyncio
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]
# Bildirimlerde Turkce ad (olmayan Ingilizce kalir)
TURKISH = {
    "person": "kişi", "cat": "kedi", "dog": "köpek", "bird": "kuş", "cup": "kupa", "bottle": "şişe",
    "chair": "sandalye", "couch": "koltuk", "bed": "yatak", "dining table": "masa", "tv": "televizyon",
    "laptop": "dizüstü bilgisayar", "mouse": "fare", "remote": "kumanda", "keyboard": "klavye",
    "cell phone": "telefon", "book": "kitap", "clock": "saat", "vase": "vazo", "scissors": "makas",
    "teddy bear": "oyuncak ayı", "backpack": "sırt çantası", "handbag": "el çantası",
    "umbrella": "şemsiye", "sports ball": "top", "bowl": "kase", "banana": "muz", "apple": "elma",
    "orange": "portakal", "potted plant": "saksı bitkisi", "wine glass": "kadeh", "spoon": "kaşık",
    "fork": "çatal", "knife": "bıçak", "refrigerator": "buzdolabı", "sink": "lavabo",
}


COLORS = ["kırmızı", "turuncu", "sarı", "yeşil", "mavi", "mor", "pembe", "kahverengi",
          "siyah", "beyaz", "gri"]
# Listede olmayan nesne: yalnizca rengiyle takip (en buyuk o renkli bolge)
COLORED = "colored object"


def turkish(name: str) -> str:
    return TURKISH.get(name, name)


def color_name(h: float, s: float, v: float) -> str:
    """HSV (0-1) -> Turkce renk adi (kaba, oda isiginda)."""
    if v < 0.22:
        return "siyah"
    if s < 0.20:
        return "beyaz" if v > 0.72 else "gri"
    deg = h * 360
    if deg < 15 or deg >= 340:
        return "kırmızı"
    if deg < 42:
        return "kahverengi" if v < 0.55 else "turuncu"
    if deg < 70:
        return "sarı"
    if deg < 165:
        return "yeşil"
    if deg < 255:
        return "mavi"
    if deg < 290:
        return "mor"
    return "pembe"


@dataclass
class Detection:
    name: str
    conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 (0-1)
    colors: tuple[str, ...] = ()  # kutu ortasindaki baskin renkler (cok olandan aza)

    @property
    def cx(self) -> float:
        return (self.box[0] + self.box[2]) / 2


class Detector:
    def __init__(self, model: Path, threads: int, threshold: float):
        self._model, self._threads, self._threshold = model, threads, threshold
        self._session = None
        self._pool = ThreadPoolExecutor(1, thread_name_prefix="vision")
        self.last: list[Detection] = []

    def _load(self):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self._threads
        self._session = ort.InferenceSession(str(self._model), opts, providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0]
        self._size = int(self._input.shape[2])
        log.info("nesne tanima modeli yuklendi: %s (%d)", self._model.name, self._size)

    def _detect(self, jpeg: bytes) -> list[Detection]:
        import numpy as np
        from PIL import Image
        if self._session is None:
            self._load()
        size = self._size
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        scale = size / max(img.width, img.height)
        w, h = round(img.width * scale), round(img.height * scale)
        px, py = (size - w) // 2, (size - h) // 2
        canvas = Image.new("RGB", (size, size), (114, 114, 114))
        canvas.paste(img.resize((w, h)), (px, py))
        x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        out = self._session.run(None, {self._input.name: x})[0][0]  # 84 x N: cx, cy, w, h, 80 sinif
        scores = out[4:]
        cls, conf = scores.argmax(0), scores.max(0)
        keep = np.where(conf >= self._threshold)[0]
        found = []
        for i in keep[np.argsort(-conf[keep])]:
            cx, cy, bw, bh = out[:4, i]
            # Kenar boslugunu cikar, goruntuye gore 0-1 oranina cevir
            box = ((cx - bw / 2 - px) / w, (cy - bh / 2 - py) / h, (cx + bw / 2 - px) / w, (cy + bh / 2 - py) / h)
            box = tuple(float(min(1.0, max(0.0, v))) for v in box)
            d = Detection(COCO80[int(cls[i])], float(conf[i]), box, _colors(img, box))
            # Ayni siniftan cakisan kutulari ele (NMS)
            if all(o.name != d.name or _iou(o.box, d.box) < 0.45 for o in found):
                found.append(d)
        return found

    def _shift(self, a: bytes, b: bytes, hfov: float) -> float:
        """Iki kare arasindaki yatay kayma (derece): faz korelasyonu, 160x120 gri."""
        import numpy as np
        from PIL import Image

        def gray(jpeg):
            g = np.asarray(Image.open(io.BytesIO(jpeg)).convert("L").resize((160, 120)), dtype=np.float32)
            return (g - g.mean()) * np.outer(np.hanning(120), np.hanning(160))
        fa, fb = np.fft.fft2(gray(a)), np.fft.fft2(gray(b))
        r = fa * np.conj(fb)
        corr = np.abs(np.fft.ifft2(r / (np.abs(r) + 1e-9)))
        dx = int(np.unravel_index(corr.argmax(), corr.shape)[1])
        dx = dx - 160 if dx > 80 else dx
        return abs(dx) / 160 * hfov

    async def shift(self, a: bytes, b: bytes, hfov: float) -> float:
        return await asyncio.get_running_loop().run_in_executor(self._pool, self._shift, a, b, hfov)

    def _change(self, a: bytes, b: bytes, ignore: list) -> float | None:
        """Iki kare ne kadar farkli: ortalama mutlak fark (0-255), 160x120 gri, parlaklik ortalamasi cikarilmis.

        ignore: haric tutulan kutular (hareket eden kisi). Kareden pek bir sey kalmazsa None.
        Pi kamerasinda yerinde duran robot 2-6, 1 derece donus ~19 verdi.
        """
        import numpy as np
        from PIL import Image

        def gray(jpeg):
            g = np.asarray(Image.open(io.BytesIO(jpeg)).convert("L").resize((160, 120)), dtype=np.float32)
            return g - g.mean()
        keep = np.ones((120, 160), bool)
        for x1, y1, x2, y2 in ignore:
            keep[int(y1 * 120):int(np.ceil(y2 * 120)), int(x1 * 160):int(np.ceil(x2 * 160))] = False
        if keep.mean() < 0.15:
            return None
        return float(np.abs(gray(a) - gray(b))[keep].mean())

    async def change(self, a: bytes, b: bytes, ignore: list) -> float | None:
        return await asyncio.get_running_loop().run_in_executor(self._pool, self._change, a, b, ignore)

    def _find_color(self, jpeg: bytes, color: str) -> list[Detection]:
        """Karedeki tek renkli bolgeler (kirmizi kutu, turuncu top...), buyukten kucuge en fazla 4; 80x60 izgarada.

        Hepsi doner: kadrajda iki kirmizi sey varsa takip onceki konuma yakin olanda kalir (yalniz en
        buyugu donunce kareden kareye ikisi arasinda gidip geliyordu)."""
        import numpy as np
        from PIL import Image
        img = Image.open(io.BytesIO(jpeg)).convert("RGB").resize((80, 60))
        hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255
        mask = _color_mask(hsv, color)
        area = mask.mean()
        if area < 0.004 or area > 0.6:  # cok kucuk (gurultu) ya da duvar/zemin
            return []
        # En buyuk bagli bolge (4-komsu, basit tasma doldurma)
        seen = np.zeros_like(mask)
        regions: list[list[tuple[int, int]]] = []
        for y, x in zip(*np.nonzero(mask)):
            if seen[y, x]:
                continue
            stack, region = [(y, x)], []
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                region.append((cy, cx))
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if 0 <= ny < 60 and 0 <= nx < 80 and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(region) >= 0.004 * 4800:
                regions.append(region)
        found = []
        for region in sorted(regions, key=len, reverse=True)[:4]:
            ys, xs = zip(*region)
            box = (float(min(xs)) / 80, float(min(ys)) / 60, float(max(xs) + 1) / 80, float(max(ys) + 1) / 60)
            found.append(Detection(COLORED, min(1.0, len(region) / 4800 * 20), box, (color,)))
        return found

    async def find_color(self, jpeg: bytes, color: str) -> list[Detection]:
        self.last = await asyncio.get_running_loop().run_in_executor(self._pool, self._find_color, jpeg, color)
        return self.last

    async def detect(self, jpeg: bytes) -> list[Detection]:
        self.last = await asyncio.get_running_loop().run_in_executor(self._pool, self._detect, jpeg)
        return self.last


def _color_mask(hsv, color: str):
    """color_name ile ayni esikler, numpy ile tum karede: renk == color olan pikseller."""
    import numpy as np
    h, s, v = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
    dark, gray = v < 0.22, (v >= 0.22) & (s < 0.20)
    chroma = (v >= 0.22) & (s >= 0.20)
    ranges = {
        "kırmızı": (h < 15) | (h >= 340), "sarı": (h >= 42) & (h < 70), "yeşil": (h >= 70) & (h < 165),
        "mavi": (h >= 165) & (h < 255), "mor": (h >= 255) & (h < 290), "pembe": (h >= 290) & (h < 340),
    }
    if color == "siyah":
        return dark
    if color == "beyaz":
        return gray & (v > 0.72)
    if color == "gri":
        return gray & (v <= 0.72)
    if color in ("turuncu", "kahverengi"):
        band = chroma & (h >= 15) & (h < 42)
        return band & (v >= 0.55) if color == "turuncu" else band & (v < 0.55)
    return chroma & ranges.get(color, np.zeros_like(dark))


def _colors(img, box) -> tuple[str, ...]:
    """Kutunun ortadaki yarisinda piksel renklerinin dagilimi: %20'den fazla olanlar."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    crop = img.crop((int((x1 + w / 4) * img.width), int((y1 + h / 4) * img.height),
                     max(int((x2 - w / 4) * img.width), int((x1 + w / 4) * img.width) + 1),
                     max(int((y2 - h / 4) * img.height), int((y1 + h / 4) * img.height) + 1)))
    crop = crop.resize((16, 16)).convert("HSV")
    counts: dict[str, int] = {}
    for hh, ss, vv in crop.getdata():
        c = color_name(hh / 255, ss / 255, vv / 255)
        counts[c] = counts.get(c, 0) + 1
    return tuple(c for c, n in sorted(counts.items(), key=lambda x: -x[1]) if n >= 0.2 * 256)


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
