"""MPU6050 jiroskop: robotun yonu (derece) ve aciyla donus.

Arka planda ~200 Hz okunur; Z ekseni dikey (modul duz takili), sola donus pozitif. Acilista
robot dururken sabit kayma (bias) olculur ve cikarilir; kalan kayma dakikada ~1-3 derece,
saniyeler suren donusler icin yeterli. Pi 4'te olculdu: bias -0.36 derece/sn, gurultu 0.04.
"""

import asyncio
import logging
import struct
import threading
import time

log = logging.getLogger(__name__)

ADDR = 0x68
PWR_MGMT_1, CONFIG, GYRO_CONFIG, GYRO_ZOUT_H, WHO_AM_I = 0x6B, 0x1A, 0x1B, 0x47, 0x75
LSB_PER_DPS = 131.0  # +-250 derece/sn araliginda


class Imu:
    def __init__(self, bus: int = 1):
        from smbus2 import SMBus
        self._bus = SMBus(bus)
        if self._bus.read_byte_data(ADDR, WHO_AM_I) != 0x68:
            raise RuntimeError("MPU6050 bulunamadi (0x68)")
        self._bus.write_byte_data(ADDR, PWR_MGMT_1, 0x00)  # uyandir
        self._bus.write_byte_data(ADDR, CONFIG, 0x03)  # dahili filtre ~44 Hz: motor titresimi
        self._bus.write_byte_data(ADDR, GYRO_CONFIG, 0x00)  # +-250 derece/sn
        self.bias = 0.0
        self.heading = 0.0  # derece, sola pozitif; acilistan beri
        self.rate = 0.0  # derece/sn
        self._lock = threading.Lock()
        self._running = False

    def _read(self) -> float:
        hi_lo = self._bus.read_i2c_block_data(ADDR, GYRO_ZOUT_H, 2)
        return struct.unpack(">h", bytes(hi_lo))[0] / LSB_PER_DPS

    def calibrate(self, seconds: float = 2.0) -> float:
        """Robot dururken: sabit kaymayi olc."""
        xs, t = [], time.monotonic()
        while time.monotonic() - t < seconds:
            xs.append(self._read())
            time.sleep(0.005)
        self.bias = sum(xs) / len(xs)
        log.info("jiroskop hazir: kayma %+.2f derece/sn", self.bias)
        return self.bias

    def start(self) -> None:
        self._running = True
        threading.Thread(target=self._loop, name="imu", daemon=True).start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        last = time.monotonic()
        while self._running:
            try:
                r = self._read() - self.bias
            except OSError:
                time.sleep(0.01)  # I2C anlik hata: bir sonraki okumada devam
                continue
            now = time.monotonic()
            with self._lock:
                self.heading += r * (now - last)
                self.rate = r
            last = now
            time.sleep(0.004)


class AngleTurner:
    """Aciyla yerinde donus: motor guclu PWM ile doner, jiroskop hedefe varinca durdurur.

    Motor durunca robot bir sure daha kayar (coast); durdurma bu payi hiza gore onceden yapar ve
    her donusten sonra olculen kaymaya gore payi gunceller. Kalan hata tolerance'tan buyukse kisa
    duzeltme donusu yapilir.
    """

    def __init__(self, imu: Imu, motors, cfg: dict):
        self._imu, self._motors, self._cfg = imu, motors, cfg
        self._coast_s = cfg["coast_s"]  # durdurma ile tam durus arasi, hiz x bu kadar derece

    async def turn(self, degrees: float) -> float:
        """degrees: sola pozitif. Hedefe kalan hatayi (derece) dondurur."""
        c = self._cfg
        target = self._imu.heading + degrees
        for attempt in range(c["max_corrections"] + 1):
            err = target - self._imu.heading
            if abs(err) <= c["tolerance_deg"]:
                break
            sign = 1 if err > 0 else -1
            duty = c["duty_fast"] if abs(err) > c["slow_below_deg"] else c["duty"]
            await self._motors.move("sol" if sign > 0 else "sag", c["max_seconds"], duty=duty, soft=False)
            while self._motors.moving:
                left = sign * (target - self._imu.heading)
                if left <= abs(self._imu.rate) * self._coast_s:
                    break
                await asyncio.sleep(0.005)
            stop_at, rate = self._imu.heading, abs(self._imu.rate)
            await self._motors.stop(quiet=True)
            await asyncio.sleep(c["settle_s"])  # kayma bitsin
            coasted = sign * (self._imu.heading - stop_at)
            if rate > 20:
                # Bir sonraki durdurma icin kayma payini ogren (yumusak: tek olcum oynak)
                self._coast_s += 0.5 * (coasted / rate - self._coast_s)
            log.info("aci donusu: hedef %+.1f, kalan %+.1f derece (durdururken %.0f derece/sn, kayma %.1f "
                     "derece, pay %.3f sn, %d. deneme)", degrees, target - self._imu.heading, rate,
                     coasted, self._coast_s, attempt + 1)
        return target - self._imu.heading
