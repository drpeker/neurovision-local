"""Acilis anonsu: Pi acilir acilmaz 3 tonlu ses ve ekranda "Neuro-Vision / Booting..".

Robot programindan bagimsiz (nv/ kullanmaz); neurovision-boot.service ile robottan once calisir.
Ekran robot baslayana kadar boyle kalir; robot kendi ekranini acinca degisir.
"""

import colorsys
import math
import os
import struct
import subprocess
import time

# lgpio bildirim borusunu calisma klasorunde acar; servis "/" klasorunde baslar, oraya yazamaz
os.chdir("/tmp")
import lgpio  # noqa: E402
import spidev
from PIL import Image, ImageChops, ImageDraw, ImageFont

W, H, Y_OFFSET = 240, 280, 20
DC, RST, BL = 25, 5, 6
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
RATE = 24000
NOTES = [(659.3, 0.0), (784.0, 0.16), (1046.5, 0.32)]  # Mi, Sol, Do: yukselen yumusak uclu


def chime() -> bytes:
    """Uc nota, ust uste binen yumusak sonumlu; 16 bit mono PCM."""
    n = int(RATE * 1.3)
    out = [0.0] * n
    for freq, start in NOTES:
        s0 = int(start * RATE)
        for i in range(n - s0):
            t = i / RATE
            env = min(1.0, t / 0.01) * math.exp(-t * 4.0)  # 10 ms giris, zil gibi sonum
            out[s0 + i] += env * (math.sin(2 * math.pi * freq * t) + 0.25 * math.sin(4 * math.pi * freq * t))
    return b"".join(struct.pack("<h", int(max(-1, min(1, v * 0.22)) * 32767)) for v in out)


def play(pcm: bytes) -> None:
    # Ses karti acilista gec hazir olabilir: birkac kez dene
    for _ in range(10):
        r = subprocess.run(["aplay", "-q", "-D", "speaker", "-t", "raw", "-f", "S16_LE",
                            "-r", str(RATE), "-c", "1"], input=pcm, capture_output=True)
        if r.returncode == 0:
            return
        time.sleep(0.5)


def screen() -> bytes:
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    big, small = ImageFont.truetype(FONT_BOLD, 34), ImageFont.truetype(FONT, 22)
    word = "Neuro-Vision"
    x = (W - d.textlength(word, font=big)) / 2
    for i, ch in enumerate(word):
        r, g, b = colorsys.hsv_to_rgb(0.52 + i * 0.035, 0.8, 1.0)
        d.text((x, H * 0.47), ch, font=big, fill=(int(r * 255), int(g * 255), int(b * 255)), anchor="ls")
        x += d.textlength(ch, font=big)
    d.text((W / 2, H * 0.60), "Booting..", font=small, fill=(200, 210, 225), anchor="ms")
    # RGB565, buyuk uclu
    r, g, b = img.split()
    hi = ImageChops.add(r.point(lambda v: v & 0xF8), g.point(lambda v: v >> 5))
    lo = ImageChops.add(g.point(lambda v: (v << 3) & 0xE0), b.point(lambda v: v >> 3))
    return Image.merge("LA", (hi, lo)).tobytes()


def show(data: bytes) -> None:
    h = lgpio.gpiochip_open(0)
    for pin, level in ((DC, 0), (RST, 1), (BL, 0)):
        lgpio.gpio_claim_output(h, pin, level)
    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.max_speed_hz, spi.mode = 40_000_000, 0

    def cmd(c: int, payload: bytes = b"") -> None:
        lgpio.gpio_write(h, DC, 0)
        spi.writebytes([c])
        if payload:
            lgpio.gpio_write(h, DC, 1)
            spi.writebytes2(payload)

    lgpio.gpio_write(h, RST, 0)
    time.sleep(0.05)
    lgpio.gpio_write(h, RST, 1)
    time.sleep(0.15)
    for c, payload, wait in ((0x01, b"", 0.15), (0x11, b"", 0.12), (0x3A, b"\x55", 0),
                             (0x36, b"\x00", 0), (0x21, b"", 0), (0x13, b"", 0), (0x29, b"", 0.05)):
        cmd(c, payload)
        time.sleep(wait)
    y0, y1 = Y_OFFSET, Y_OFFSET + H - 1
    cmd(0x2A, bytes([0, 0, 0, W - 1]))
    cmd(0x2B, bytes([y0 >> 8, y0 & 255, y1 >> 8, y1 & 255]))
    cmd(0x2C, data)
    # Arka isik sabit acik (PWM degil): program cikinca da acik kalir
    lgpio.gpio_write(h, BL, 1)
    spi.close()
    lgpio.gpiochip_close(h)


if __name__ == "__main__":
    pcm = chime()
    show(screen())
    play(pcm)
