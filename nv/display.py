"""Robotun yuzu: ST7789 SPI ekranda gozler, durum, duygu ve altta altyazi.

Yuz cizimi saf Python: her satir icin "acik araliklar" (span) hesaplanir ve bytearray dilimine
tek seferde yazilir. Altyazi PIL ile yalnizca metin degisince cizilir. Sadece degisen satirlar
gonderilir, SPI yazimi ayri bir is parcaciginda (ses akisi beklemez). Hareket yokken kare cizilmez.
"""

import asyncio
import logging
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

log = logging.getLogger(__name__)

EMOTIONS = ["nötr", "mutlu", "üzgün", "şaşkın", "kızgın", "uykulu"]


def rgb(r: int, g: int, b: int) -> bytes:
    v = (r & 0xF8) << 8 | (g & 0xFC) << 3 | b >> 3
    return bytes([v >> 8, v & 255])


BLACK = rgb(0, 0, 0)
COLORS = {
    "eye": (40, 210, 255), "listen": (60, 255, 210), "offline": (90, 90, 100),
    "hot": (255, 150, 30), "flash": (255, 255, 255), "accent": (40, 210, 255),
    "ok": (40, 220, 90), "warn": (255, 200, 0), "bad": (255, 50, 40),
    "user": (150, 150, 165), "robot": (225, 240, 255), "radio": (40, 210, 255),
}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def to_rgb565(img) -> bytes:
    """PIL RGB goruntuyu panelin RGB565 (buyuk uclu) baytlarina cevir; dongu PIL icinde (C)."""
    from PIL import Image, ImageChops
    r, g, b = img.split()
    hi = ImageChops.add(r.point(lambda v: v & 0xF8), g.point(lambda v: v >> 5))
    lo = ImageChops.add(g.point(lambda v: (v << 3) & 0xE0), b.point(lambda v: v >> 3))
    return Image.merge("LA", (hi, lo)).tobytes()


def camera_preview(jpeg: bytes, w: int, h: int) -> bytes:
    """Kamera karesini w x h alana oranini koruyarak sigdir, ortala (RGB565)."""
    import io

    from PIL import Image
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    img.thumbnail((w, h))
    out = Image.new("RGB", (w, h))
    out.paste(img, ((w - img.width) // 2, (h - img.height) // 2))
    return to_rgb565(out)


def splash(w: int, h: int) -> bytes:
    """Acilis ekrani: renkli "Peker", altinda "AI Technology" (RGB565)."""
    import colorsys

    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)
    big = ImageFont.truetype(FONT_BOLD, 58)
    small = ImageFont.truetype(FONT, 22)
    word = "Peker"
    x = (w - draw.textlength(word, font=big)) / 2
    for i, ch in enumerate(word):
        # Harf harf gok kusagi: maviden mora, turuncuya
        r, g, b = colorsys.hsv_to_rgb(0.55 + i * 0.12, 0.85, 1.0)
        draw.text((x, h * 0.47), ch, font=big, fill=(int(r * 255), int(g * 255), int(b * 255)),
                  anchor="ls")
        x += draw.textlength(ch, font=big)
    draw.line((w * 0.2, h * 0.54, w * 0.8, h * 0.54), fill=(40, 210, 255), width=2)
    draw.text((w / 2, h * 0.63), "AI Technology", font=small, fill=(225, 240, 255), anchor="ms")
    return to_rgb565(img)


class Panel:
    """ST7789 dusuk seviye: baslatma, satir blokla yazma, arka isik PWM."""

    def __init__(self, cfg: dict):
        import lgpio
        import spidev
        self._lg = lgpio
        self.w, self.h, self._yoff = cfg["width"], cfg["height"], cfg["y_offset"]
        self._dc, self._rst, self._bl = cfg["dc_pin"], cfg["rst_pin"], cfg["backlight_pin"]
        self._gpio = lgpio.gpiochip_open(0)
        for pin, level in ((self._dc, 0), (self._rst, 1), (self._bl, 0)):
            lgpio.gpio_claim_output(self._gpio, pin, level)
        self._spi = spidev.SpiDev()
        self._spi.open(cfg["spi_bus"], cfg["spi_device"])
        self._spi.max_speed_hz = cfg["spi_hz"]
        self._spi.mode = 0
        lgpio.gpio_write(self._gpio, self._rst, 0)
        time.sleep(0.05)
        lgpio.gpio_write(self._gpio, self._rst, 1)
        time.sleep(0.15)
        for c, data, wait in ((0x01, b"", 0.15), (0x11, b"", 0.12), (0x3A, b"\x55", 0),
                              (0x36, b"\x00", 0), (0x21, b"", 0), (0x13, b"", 0), (0x29, b"", 0.05)):
            self._cmd(c, data)
            time.sleep(wait)

    def _cmd(self, c: int, data: bytes = b"") -> None:
        self._lg.gpio_write(self._gpio, self._dc, 0)
        self._spi.writebytes([c])
        if data:
            self._lg.gpio_write(self._gpio, self._dc, 1)
            self._spi.writebytes2(data)

    def blit(self, buf: bytearray, y0: int, y1: int) -> None:
        """buf'taki y0..y1 (dahil) satirlarini tam genislikte gonder."""
        x1, ya, yb = self.w - 1, y0 + self._yoff, y1 + self._yoff
        self._cmd(0x2A, bytes([0, 0, x1 >> 8, x1 & 255]))
        self._cmd(0x2B, bytes([ya >> 8, ya & 255, yb >> 8, yb & 255]))
        self._cmd(0x2C)
        self._lg.gpio_write(self._gpio, self._dc, 1)
        row = self.w * 2
        self._spi.writebytes2(memoryview(buf)[y0 * row:(y1 + 1) * row])

    def backlight(self, percent: int) -> None:
        # PWM degil, acik/kapali: ayni anda calisan ikinci bir lgpio yazilim PWM'i (arka isik)
        # motor PWM'ini ~30 darbe sonra tamamen durduruyordu (lgpio "calisiyor" diyordu).
        # Kisma icin donanim PWM'li bir pin (GPIO12/13) gerekir.
        self._lg.gpio_write(self._gpio, self._bl, 1 if percent > 0 else 0)

    def close(self) -> None:
        self.blit(bytearray(self.w * self.h * 2), 0, self.h - 1)
        self.backlight(0)
        self._spi.close()
        self._lg.gpiochip_close(self._gpio)


class Canvas:
    """Satir araliklariyla cizim: her sekil satir satir [x0, x1) araliklari uretir."""

    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.buf = bytearray(w * h * 2)

    def clear_rows(self, y0: int, y1: int) -> None:
        self.buf[y0 * self.w * 2:(y1 + 1) * self.w * 2] = bytes((y1 - y0 + 1) * self.w * 2)

    def span(self, y: int, x0: float, x1: float, color: bytes) -> None:
        a, b = max(0, int(x0)), min(self.w, int(x1))
        if 0 <= y < self.h and b > a:
            i = (y * self.w + a) * 2
            self.buf[i:i + (b - a) * 2] = color * (b - a)

    def circle(self, cx: float, cy: float, r: float, color: bytes) -> None:
        for y in range(int(cy - r), int(cy + r) + 1):
            d = r * r - (y - cy) ** 2
            if d > 0:
                half = math.sqrt(d)
                self.span(y, cx - half, cx + half, color)

    def rounded_rect(self, cx: float, cy: float, w: float, h: float, color: bytes,
                     cut=None) -> None:
        """cut(y, x0, x1) -> [(x0, x1), ...] : goz kapagi gibi kesimler icin."""
        top, bottom = int(cy - h / 2), int(cy + h / 2)
        r = min(w, h) * 0.32
        for y in range(top, bottom):
            d = min(y - top, bottom - 1 - y)
            inset = r - math.sqrt(max(0.0, r * r - (r - d) ** 2)) if d < r else 0.0
            x0, x1 = cx - w / 2 + inset, cx + w / 2 - inset
            for a, b in (cut(y, x0, x1) if cut else [(x0, x1)]):
                self.span(y, a, b, color)


class Captions:
    """Alt seritte son konusma: kullanici gri, robot beyaz. Sigmazsa son satirlar gorunur."""

    def __init__(self, w: int, lines: int, font_px: int):
        from PIL import ImageFont
        try:
            self._font = ImageFont.truetype(FONT, font_px)
        except OSError:
            self._font = ImageFont.load_default(font_px)
        self.w, self.lines, self.line_h = w, lines, font_px + 5
        self.h = lines * self.line_h

    def _wrap(self, text: str) -> list[str]:
        out, line = [], ""
        for word in text.split():
            test = f"{line} {word}" if line else word
            if self._font.getlength(test) <= self.w - 12:
                line = test
            else:
                if line:
                    out.append(line)
                line = word
        if line:
            out.append(line)
        return out[-self.lines:]

    def render(self, text: str, color: tuple[int, int, int]) -> bytes:
        """Metni seritin RGB565 baytlarina cevir (is parcaciginda calisir)."""
        from PIL import Image, ImageDraw
        img = Image.new("L", (self.w, self.h))
        draw = ImageDraw.Draw(img)
        for i, line in enumerate(self._wrap(text)):
            draw.text((self.w / 2, i * self.line_h + 2), line, font=self._font, fill=255, anchor="ma")
        # Kenar yumusatma: gri seviyeyi renge olcekle
        lut = [rgb(*(c * v // 255 for c in color)) for v in range(256)]
        return b"".join(lut[v] for v in img.tobytes())


class Face:
    """Durumu robotun parcalarindan okur, yuzu cizer. LiveBrain yalnizca thinking/flash/emotion verir."""

    def __init__(self, cfg: dict, panel: Panel, speaker, gate, body):
        self._cfg = cfg
        self._panel = panel
        self._speaker, self._gate, self._body = speaker, gate, body
        self._canvas = Canvas(panel.w, panel.h)
        self._pool = ThreadPoolExecutor(1, thread_name_prefix="display")
        w, h = panel.w, panel.h
        self._captions = (Captions(w, cfg["caption_lines"], cfg["caption_font_px"])
                          if cfg.get("captions") else None)
        # Altyazi varsa yuz yukari kayar, altta serit acilir
        self._cap_top = h - self._captions.h - 4 if self._captions else h
        self._face_y = (34, self._cap_top - 1) if self._captions else (40, h - 30)
        self._cx, self._cy = w / 2, h * (0.40 if self._captions else 0.47)
        self._eye_w, self._eye_h, self._gap = w * 0.25, h * 0.27, w * 0.15
        # yumusak gecis yapilan parametreler: bakis, olcek, goz acikligi
        self._cur = {"dx": 0.0, "dy": 0.0, "sw": 1.0, "sh": 1.0}
        self._tgt = dict(self._cur)
        self._blink_at = time.monotonic() + 3
        self._blink_t = -1.0
        self._look_at = time.monotonic() + 2
        self._flash_until = 0.0
        self._emotion, self._emotion_until = "nötr", 0.0
        self._thinking_since: float | None = None
        self._last_active = time.monotonic()
        self._bright = None
        self._temp, self._temp_at = 0.0, 0.0
        self._status_key = None
        self._caption = ("", "user")
        self._caption_at = 0.0
        self._caption_shown = None
        self._caption_drawn_at = 0.0
        self.radio = None  # robot.py baglar: altyazi yokken calan istasyon gorunur
        self._gaze: float | None = None  # takip: hedefin yatay konumu (-0.5..0.5)
        self._preview: bytes | None = None  # look karesi: yuz yerine kisa sure gosterilir
        self._preview_until = 0.0
        self._task = None

    # --- LiveBrain'den ---
    def flash(self) -> None:
        self._flash_until = time.monotonic() + 0.35
        self.poke()

    def set_emotion(self, emotion: str, seconds: float = 5.0) -> None:
        self._emotion = emotion if emotion in EMOTIONS else "nötr"
        self._emotion_until = time.monotonic() + seconds
        self.poke()

    def poke(self) -> None:
        self._last_active = time.monotonic()

    @property
    def thinking(self) -> bool:
        """Cevap bekleniyor. Gelmezse (gurultu modele gitti, model sustu) thinking_timeout_s
        sonra kendiliginden biter; yoksa yuz bir sonraki konusmaya kadar dusunur gorunuyordu."""
        if self._thinking_since is None:
            return False
        if time.monotonic() - self._thinking_since > self._cfg["thinking_timeout_s"]:
            self._thinking_since = None
            return False
        return True

    @thinking.setter
    def thinking(self, value: bool) -> None:
        self._thinking_since = time.monotonic() if value else None

    def gaze(self, offset: float | None) -> None:
        """Takipte gozler hedefin yonune baksin; None: serbest."""
        self._gaze = offset
        self.poke()

    def preview(self, jpeg: bytes) -> None:
        """Robotun baktigi kare, yuzun yerinde preview_s saniye (tek kare, canli degil)."""
        self._preview = jpeg
        self.poke()

    def caption(self, text: str, who: str) -> None:
        """who: "user" ya da "robot". Metin akarken her parcada tamami verilir."""
        self._caption = (text.strip(), who)
        self._caption_at = time.monotonic()

    # --- dongu ---
    async def start(self) -> None:
        w, h = self._panel.w, self._panel.h
        logo = await asyncio.get_running_loop().run_in_executor(self._pool, splash, w, h)
        await self._run_panel(self._panel.blit, bytearray(logo), 0, h - 1)
        await self._backlight(time.monotonic())
        self._task = asyncio.create_task(self._loop(), name="display")
        log.info("ekran hazir: %dx%d", self._panel.w, self._panel.h)

    async def _run_panel(self, fn, *args) -> None:
        await asyncio.get_running_loop().run_in_executor(self._pool, fn, *args)

    def _state(self, now: float) -> str:
        if self._body.connected_since is None:
            return "offline"
        if now < self._flash_until:
            return "looking"
        if self._speaker.busy:
            return "speaking"
        if self._gate.active:
            return "listening"
        if self.thinking:
            return "thinking"
        return "idle"

    async def _loop(self) -> None:
        # Acilis ekrani kalsin (robot bu arada ortami dinleyip baglaniyor), sonra yuz
        await asyncio.sleep(self._cfg["splash_s"])
        await self._run_panel(self._panel.blit, self._canvas.buf, 0, self._panel.h - 1)
        self._status_key = None
        while True:
            try:
                await self._frames()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Yuz hatasi robotu durdurmasin; kaydet, kisa bekle, devam et
                log.exception("ekran dongusu hatasi")
                self._status_key = None
                await asyncio.sleep(1)

    async def _frames(self) -> None:
        last_frame = None
        while True:
            now = time.monotonic()
            state = self._state(now)
            if state != "idle":
                self._last_active = now
            await self._backlight(now)
            await self._status(now)
            await self._draw_caption(now)
            if await self._draw_preview(now):
                last_frame = None  # kare bitince yuz bastan cizilsin
                await asyncio.sleep(0.2)
                continue
            self._update_targets(state, now)
            moving = self._tween()
            animated = state in ("speaking", "thinking", "listening", "looking") or self._blink_t >= 0
            if moving or animated or last_frame != (state, self._emotion):
                y0, y1 = self._draw(state, now)
                await self._run_panel(self._panel.blit, self._canvas.buf, y0, y1)
                last_frame = (state, self._emotion)
            await asyncio.sleep(0.05 if (moving or animated) else 0.2)

    def _update_targets(self, state: str, now: float) -> None:
        if now > self._emotion_until:
            self._emotion = "nötr"
        if self._gaze is not None:
            self._tgt["dx"], self._tgt["dy"] = self._gaze * 50, 0.0
        elif state == "idle" and now > self._look_at:
            self._tgt["dx"] = random.uniform(-22, 22)
            self._tgt["dy"] = random.uniform(-10, 10)
            self._look_at = now + random.uniform(2.0, 5.0)
        elif state == "thinking":
            self._tgt["dx"], self._tgt["dy"] = 12.0, -14.0
        elif state in ("listening", "speaking"):
            self._tgt["dx"], self._tgt["dy"] = 0.0, 0.0
        big = {"listening": 1.08, "looking": 1.15}.get(state, 1.0)
        emo = self._emotion
        self._tgt["sw"] = big * (1.12 if emo == "şaşkın" else 1.0)
        self._tgt["sh"] = big * {"şaşkın": 1.25, "uykulu": 0.35}.get(emo, 1.0) * (0.12 if state == "offline" else 1.0)
        # goz kirpma: 0.18 sn'de kapanip acilir
        if self._blink_t < 0 and now > self._blink_at and state in ("idle", "listening"):
            self._blink_t = now
        if self._blink_t >= 0 and now - self._blink_t > 0.18:
            self._blink_t = -1.0
            self._blink_at = now + random.uniform(2.5, 6.0)

    def _tween(self) -> bool:
        moving = False
        for k, v in self._tgt.items():
            d = v - self._cur[k]
            if abs(d) > 0.01:
                self._cur[k] += d * 0.35
                moving = True
            else:
                self._cur[k] = v
        return moving

    def _cut(self, emo: str, left: bool, top: float, h: float):
        if emo in ("üzgün", "kızgın"):
            # ust kapak egik: uzgunde dis kose, kizginda ic kose iner
            outer_side = left if emo == "üzgün" else not left

            def cut(y, x0, x1):
                k = (y - top) / (h * 0.42)
                if k >= 1:
                    return [(x0, x1)]
                drop = (1 - k) * (x1 - x0)
                return [(x0 + drop, x1)] if outer_side else [(x0, x1 - drop)]
            return cut
        if emo == "mutlu":
            # alt kisim yay seklinde silinir: ^ ^
            cy, ry = top + h * 1.05, h * 0.62

            def cut(y, x0, x1):
                dy = y - cy
                if abs(dy) >= ry:
                    return [(x0, x1)]
                half = (x1 - x0) * 0.62 * math.sqrt(1 - (dy / ry) ** 2)
                mid = (x0 + x1) / 2
                return [(x0, mid - half), (mid + half, x1)]
            return cut
        return None

    def _draw(self, state: str, now: float) -> tuple[int, int]:
        c, cur = self._canvas, self._cur
        y0, y1 = self._face_y
        c.clear_rows(y0, y1)
        hot = self._temp >= self._cfg["hot_c"]
        name = {"offline": "offline", "looking": "flash", "listening": "listen"}.get(state, "eye")
        color = rgb(*COLORS["hot" if hot and state != "offline" else name])
        open_k = 1.0
        if self._blink_t >= 0:
            p = (now - self._blink_t) / 0.18
            open_k = max(0.08, abs(1 - 2 * p))
        w, h = self._eye_w * cur["sw"], self._eye_h * cur["sh"] * open_k
        cy = self._cy + cur["dy"]
        for left in (True, False):
            cx = self._cx + cur["dx"] + (-1 if left else 1) * (self._gap / 2 + self._eye_w / 2)
            cut = self._cut(self._emotion, left, cy - h / 2, h) if h > 10 else None
            c.rounded_rect(cx, cy, w, h, color, cut)
        mouth_y = self._cy + self._eye_h * 0.5 + 34
        accent = rgb(*COLORS["accent"])
        if state == "speaking":
            mh = random.choice((4, 8, 12, 16, 20))
            c.rounded_rect(self._cx, mouth_y, 56, mh, accent)
        elif state == "thinking":
            for i in range(3):
                phase = (now * 3 - i * 0.4) % 2
                c.circle(self._cx - 22 + i * 22, mouth_y - (5 if phase < 0.5 else 0), 5, accent)
        elif state == "listening":
            c.circle(self._cx, mouth_y, 4 + 3 * abs(math.sin(now * 4)), rgb(*COLORS["listen"]))
        return y0, y1

    async def _status(self, now: float) -> None:
        if now - self._temp_at > 10:
            try:
                self._temp = int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
            except (OSError, ValueError):
                pass
            self._temp_at = now
        online = self._body.connected_since is not None
        level = "ok" if self._temp < 65 else "warn" if self._temp < self._cfg["hot_c"] else "bad"
        key = (online, level)
        if key == self._status_key:
            return
        self._status_key = key
        c = self._canvas
        c.clear_rows(8, 28)
        c.circle(36, 18, 6, rgb(*COLORS["ok" if online else "bad"]))
        c.circle(self._panel.w - 36, 18, 6, rgb(*COLORS[level]))
        await self._run_panel(self._panel.blit, c.buf, 8, 28)

    async def _draw_preview(self, now: float) -> bool:
        """Yeni kare geldiyse ciz; gosterim suresi boyunca True (yuz cizilmez)."""
        if self._preview:
            jpeg, self._preview = self._preview, None
            y0, y1 = self._face_y
            h = y1 - y0 + 1
            data = await asyncio.get_running_loop().run_in_executor(
                self._pool, camera_preview, jpeg, self._panel.w, h)
            row = self._panel.w * 2
            self._canvas.buf[y0 * row:y0 * row + len(data)] = data
            await self._run_panel(self._panel.blit, self._canvas.buf, y0, y1)
            self._preview_until = now + self._cfg["preview_s"]
        return now < self._preview_until

    async def _draw_caption(self, now: float) -> None:
        if not self._captions:
            return
        text, who = self._caption
        if text and now - self._caption_at > self._cfg["caption_hold_s"]:
            text = ""
        if not text and self.radio and self.radio.playing:
            text, who = f"♪ {self.radio.playing}", "radio"
        # Akan metinde her parcada degil, en fazla 0.15 sn'de bir ciz
        if (text, who) == self._caption_shown or now - self._caption_drawn_at < 0.15:
            return
        self._caption_shown, self._caption_drawn_at = (text, who), now
        data = await asyncio.get_running_loop().run_in_executor(
            self._pool, self._captions.render, text, COLORS[who])
        row = self._panel.w * 2
        self._canvas.buf[self._cap_top * row:self._cap_top * row + len(data)] = data
        await self._run_panel(self._panel.blit, self._canvas.buf, self._cap_top,
                              self._cap_top + self._captions.h - 1)

    async def _backlight(self, now: float) -> None:
        want = (self._cfg["bright"] if now - self._last_active < self._cfg["dim_after_s"]
                else self._cfg["dim"])
        if want != self._bright:
            self._bright = want
            await self._run_panel(self._panel.backlight, want)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
        await self._run_panel(self._panel.close)
        self._pool.shutdown(wait=False)
