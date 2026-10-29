"""HC-SR04 ultrasonik mesafe sensoru: robotun onundeki engel (cm).

Arka planda saniyede ~14 olcum. Yanki suresi lgpio alert'leriyle (cekirdek zaman damgasi, ns)
olculur, Python'un zamanlamasina bagli degil. ECHO 5 V: 1k/2k direnc bolucuyle 3.3 V'a indirilir.
Pi 4'te olculdu (metreyle): 25 -> 23.8, 49.5 -> 47.3, 200 -> 201.4, 240 -> 238.4 cm; ara sira tek
basina yanlis bir yanki gelir (200 cm'de bir kez 125), son 3 olcumun ortancasi onu atar.
"""

import logging
import statistics
import threading
import time

log = logging.getLogger(__name__)

SOUND_CM_S = 34340.0  # ~20 C


class Sonar:
    def __init__(self, cfg: dict):
        import lgpio
        self._lg = lgpio
        self._cfg = cfg
        self._trig, self._echo = cfg["trig"], cfg["echo"]
        self.max_cm = cfg["max_cm"]
        self._h = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_output(self._h, self._trig, 0)
        lgpio.gpio_claim_alert(self._h, self._echo, lgpio.BOTH_EDGES)
        self._rise = 0
        self._last: float | None = None  # son yankinin mesafesi (cm); yanki yoksa None
        self._done = threading.Event()
        self._cb = lgpio.callback(self._h, self._echo, lgpio.BOTH_EDGES, self._edge)
        self._recent: list[tuple[float, float | None]] = []  # (zaman, cm ya da None)
        self._lock = threading.Lock()
        self._running = False

    def _edge(self, chip, gpio, level, tick) -> None:
        if level == 1:
            self._rise = tick
        elif level == 0 and self._rise:
            self._last = (tick - self._rise) / 1e9 * SOUND_CM_S / 2
            self._rise = 0
            self._done.set()

    def _ping(self) -> float | None:
        self._last, self._rise = None, 0
        self._done.clear()
        self._lg.gpio_write(self._h, self._trig, 1)
        time.sleep(0.00001)
        self._lg.gpio_write(self._h, self._trig, 0)
        self._done.wait(0.03)  # 4 m ~ 23 ms; yanki yoksa sensor ~38 ms sonra birakir
        d = self._last
        return d if d is not None and 2 <= d <= self.max_cm else None

    def start(self) -> None:
        self._running = True
        threading.Thread(target=self._loop, name="sonar", daemon=True).start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        while self._running:
            d = self._ping()
            with self._lock:
                self._recent = (self._recent + [(time.monotonic(), d)])[-3:]
            time.sleep(self._cfg["interval_s"])  # onceki darbenin yankilari sonsun
        self._cb.cancel()
        self._lg.gpiochip_close(self._h)

    @property
    def distance(self) -> float | None:
        """Son 3 olcumun ortancasi (cm); onunde max_cm icinde bir sey yoksa ya da olcum eskiyse None."""
        with self._lock:
            recent = list(self._recent)
        if len(recent) < 3 or time.monotonic() - recent[0][0] > 1.0:
            return None
        found = [d for _, d in recent if d is not None]
        return statistics.median(found) if len(found) >= 2 else None
