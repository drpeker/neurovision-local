"""ST7789 1.69" 240x280 test: kirmizi, yesil, mavi, sonra renk cubuklari + beyaz cerceve.

Pinler: SPI0 CE0 (GPIO8), DC GPIO25, RST GPIO5, BL GPIO6. Panel bellegi 240x320, gorunen
alan 240x280: satir kaydirmasi 20.
"""
import sys
import time

import lgpio
import spidev

DC, RST, BL = 25, 5, 6
W, H, Y_OFF = 240, 280, int(sys.argv[1]) if len(sys.argv) > 1 else 20

gpio = lgpio.gpiochip_open(0)
for pin, level in ((DC, 0), (RST, 1), (BL, 0)):
    lgpio.gpio_claim_output(gpio, pin, level)
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 40_000_000
spi.mode = 0


def cmd(c, data=b""):
    lgpio.gpio_write(gpio, DC, 0)
    spi.writebytes([c])
    if data:
        lgpio.gpio_write(gpio, DC, 1)
        spi.writebytes2(data)


def window(x0, y0, x1, y1):
    cmd(0x2A, bytes([x0 >> 8, x0 & 255, x1 >> 8, x1 & 255]))
    y0, y1 = y0 + Y_OFF, y1 + Y_OFF
    cmd(0x2B, bytes([y0 >> 8, y0 & 255, y1 >> 8, y1 & 255]))
    cmd(0x2C)


def rgb565(r, g, b):
    v = (r & 0xF8) << 8 | (g & 0xFC) << 3 | b >> 3
    return bytes([v >> 8, v & 255])


def fill(color, x0=0, y0=0, x1=W - 1, y1=H - 1):
    window(x0, y0, x1, y1)
    lgpio.gpio_write(gpio, DC, 1)
    spi.writebytes2(color * ((x1 - x0 + 1) * (y1 - y0 + 1)))


# Donanim reset + baslatma
lgpio.gpio_write(gpio, RST, 0); time.sleep(0.05)
lgpio.gpio_write(gpio, RST, 1); time.sleep(0.15)
cmd(0x01); time.sleep(0.15)          # SWRESET
cmd(0x11); time.sleep(0.12)          # SLPOUT
cmd(0x3A, b"\x55")                   # COLMOD: 16 bit
cmd(0x36, b"\x00")                   # MADCTL: dikey, RGB
cmd(0x21)                            # INVON: IPS paneller ters renk ister
cmd(0x13)                            # NORON
cmd(0x29); time.sleep(0.05)          # DISPON
lgpio.gpio_write(gpio, BL, 1)        # arka isik

t0 = time.time()
for name, c in (("kirmizi", rgb565(255, 0, 0)), ("yesil", rgb565(0, 255, 0)), ("mavi", rgb565(0, 0, 255))):
    t = time.time()
    fill(c)
    print(f"{name}: {(time.time() - t) * 1000:.0f} ms")
    time.sleep(1.2)

# Renk cubuklari + 2 piksel beyaz cerceve (dort kenar gorunmeli)
bars = [(255, 255, 255), (255, 255, 0), (0, 255, 255), (0, 255, 0), (255, 0, 255), (255, 0, 0), (0, 0, 255),
        (0, 0, 0)]
bw = W // len(bars)
for i, c in enumerate(bars):
    fill(rgb565(*c), i * bw, 0, min(W - 1, (i + 1) * bw - 1), H - 1)
white = rgb565(255, 255, 255)
fill(white, 0, 0, W - 1, 1); fill(white, 0, H - 2, W - 1, H - 1)
fill(white, 0, 0, 1, H - 1); fill(white, W - 2, 0, W - 1, H - 1)
fill(rgb565(255, 0, 0), W // 2 - 10, H // 2 - 10, W // 2 + 9, H // 2 + 9)  # ortada kirmizi kare
print("cubuklar + cerceve hazir; tam ekran yenileme ~", f"{(time.time() - t0):.1f} sn toplam")
spi.close()
lgpio.gpiochip_close(gpio)
