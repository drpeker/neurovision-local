"""Palet motorlari (MX1508 ya da L298N). Guvenlik kurallari burada, modelin kararina birakilmaz.

MX1508'de EN yok: hiz, yone gore IN1 ya da IN2'ye verilen PWM ile ayarlanir, digeri 0.
L298N'de yon IN'lerde, PWM EN'de. Tum hizlar hiz duzeyiyle (%, sesle ayarlanir) carpilir.

- her hareket en fazla max_seconds; yeni komut eskisini iptal eder
- yon degisirken once dur, kisa bekle
- yumusak kalkis (ramp_start'tan hedefe ramp_up_ms) ve yumusak durus (ramp_down_ms);
  stop komutu, baglanti kopmasi ve cikista ise ANINDA durur
- sicak ya da dusuk voltajda hiz dusurulur
- motor calisirken (ve durduktan sonra mute_tail_ms) mikrofon susar: motor sesi modele gitmez
- ileri giderken sonar onde stop_cm'den yakin engel gorurse ANINDA durur; engel varken ileri gitmez
"""

import asyncio
import logging
import subprocess
import time
from pathlib import Path

log = logging.getLogger(__name__)

# yon -> (sol, sag) palet yonu
MOVES = {"ileri": (1, 1), "geri": (-1, -1), "sol": (-1, 1), "sag": (1, -1)}
SPEEDS = {"çok yavaş": 0.6, "yavaş": 0.8, "normal": 1.0, "hızlı": 1.25}
RAMP_STEP_S = 0.025  # rampada guc guncelleme araligi


class Obstacle(Exception):
    """Ileri giderken onde engel: hareket aninda kesilir."""


class Motors:
    def __init__(self, cfg: dict):
        import lgpio
        self._lg = lgpio
        self._cfg = cfg
        self._h = lgpio.gpiochip_open(0)
        self._sides = {"sol": cfg["left"], "sag": cfg["right"]}
        self._mx = cfg["driver"] == "mx1508"
        for side in self._sides.values():
            for key in ("in1", "in2") if self._mx else ("in1", "in2", "en"):
                lgpio.gpio_claim_output(self._h, side[key], 0)
        self._pwm_pin: dict[int, int | None] = {}  # MX1508: taraf (in1) -> PWM verilen pin
        self.speed_level = cfg["speed_level"]
        self._derate_k = 1.0  # surekli surusta her komutta sicaklik olculmez; move() gunceller
        self._task: asyncio.Task | None = None
        self._prev_dir: tuple[int, int] = (0, 0)  # son hareketin yonu (durunca sifirlanmaz)
        self._stopped_at = 0.0
        self._moving_until = 0.0
        self._quiet_until = 0.0
        self.imu = None  # robot.py: nv.imu.Imu varsa duz gidiste yon korunur
        self._steer = 0.0  # duz gidiste yon duzeltmesi: sol guc x (1+steer), sag x (1-steer)
        self.sonar = None  # robot.py: nv.sonar.Sonar varsa ileri giderken engelde durur
        self.stop_cm = 25.0
        self.on_obstacle = None  # robot.py: on_obstacle(cm), engel yuzunden durunca (soylemek icin)
        self._drive_cmd: list | None = None  # surekli surus: [yonler, guc, yon duzeltmesi, bitis]
        log.info("motorlar hazir: %s, hiz duzeyi %%%d (sol A, sag B%s)", cfg["driver"],
                 self.speed_level, " ters" if cfg["right"]["invert"] else "")

    @property
    def speed_level(self) -> int:
        return self._level

    @speed_level.setter
    def speed_level(self, percent: int) -> None:
        self._level = max(20, min(100, int(percent)))

    def duty(self, direction: str, speed: str = "normal") -> float:
        """Hedef PWM (%): taban hiz x hiz secenegi x hiz duzeyi, tavan ve taban sinirli."""
        c = self._cfg
        base = c["turn_speed"] if direction in ("sol", "sag") else c["drive_speed"]
        duty = min(c["max_speed"], base * SPEEDS.get(speed, 1.0) * self._level / 100)
        return max(duty, c["min_speed"])

    def status(self) -> dict:
        return {"sürücü": self._cfg["driver"].upper(), "hız_düzeyi_%": self._level,
                "hareket_ediyor": self.busy,
                "düz_gidiş_gücü_%": round(self.duty("ileri")), "dönüş_gücü_%": round(self.duty("sol")),
                "not": "Hız düzeyi tüm hareketleri çarpar; yavaş/hızlı seçenekleri bunun üstüne uygulanır."}

    @property
    def moving(self) -> bool:
        """Motorlara guc gidiyor (busy'den farkli olarak susma payi yok): takip durunca hemen bakar."""
        return time.monotonic() < self._moving_until

    @property
    def busy(self) -> bool:
        """Motor calisiyor ya da yeni durdu: mikrofon bu surede susar."""
        return time.monotonic() < max(self._moving_until, self._quiet_until)

    # --- dusuk seviye ---
    def _free(self, pin: int) -> None:
        # tx_pwm(..., 0) arka planda calismaya devam edip sonraki yazmalari eziyor: pini birak
        self._lg.gpio_free(self._h, pin)
        self._lg.gpio_claim_output(self._h, pin, 0)

    def _drive_mx(self, side: dict, direction: int, duty: float) -> None:
        a, b = side["in1"], side["in2"]
        if direction == 0 or duty <= 0:
            self._free(a)
            self._free(b)
            self._pwm_pin[a] = None
            return
        if side["invert"]:
            direction = -direction
        on = a if direction > 0 else b
        if self._pwm_pin.get(a) != on:
            # Yon degisti ya da yeni basliyor: iki pin de temiz 0'dan baslasin
            self._free(a)
            self._free(b)
            self._pwm_pin[a] = on
        self._lg.tx_pwm(self._h, on, self._cfg["pwm_hz"], min(100.0, duty))

    def _drive(self, side: dict, direction: int, duty: float) -> None:
        if self._mx:
            self._drive_mx(side, direction, duty)
            return
        if direction == 0 or duty <= 0:
            # tx_pwm(..., 0) arka planda calismaya devam edip sonraki yazmalari eziyor: pini birak
            self._lg.gpio_free(self._h, side["en"])
            self._lg.gpio_claim_output(self._h, side["en"], 0)
            self._lg.gpio_write(self._h, side["in1"], 0)
            self._lg.gpio_write(self._h, side["in2"], 0)
            return
        if side["invert"]:
            direction = -direction
        self._lg.gpio_write(self._h, side["in1"], 1 if direction > 0 else 0)
        self._lg.gpio_write(self._h, side["in2"], 0 if direction > 0 else 1)
        self._lg.tx_pwm(self._h, side["en"], self._cfg["pwm_hz"], min(100.0, duty))

    def _set(self, left: int, right: int, duty: float) -> None:
        # trim: motorlar ayni gucte ayni hizda donmuyor; yavas tarafin gucu bu oranla carpilir
        for name, direction, k in (("sol", left, 1 + self._steer), ("sag", right, 1 - self._steer)):
            side = self._sides[name]
            self._drive(side, direction, duty * side.get("trim", 1.0) * k)

    def halt(self) -> None:
        """Senkron, aninda dur (cikis, hata, baglanti kopmasi)."""
        self._set(0, 0, 0)
        self._stopped_at = time.monotonic()
        self._moving_until = 0.0
        self._quiet_until = self._stopped_at + self._cfg["mute_tail_ms"] / 1000

    # --- hiz siniri ---
    def ahead(self) -> float | None:
        """Ondeki engel stop_cm'den yakinsa mesafesi (cm), degilse None."""
        d = self.sonar.distance if self.sonar else None
        return d if d is not None and d < self.stop_cm else None

    def _check(self, dirs: tuple[int, int]) -> None:
        if dirs == MOVES["ileri"] and self.ahead() is not None:
            raise Obstacle()

    def _derate(self) -> float:
        """Sicaklik ya da dusuk voltaj varsa hizi dusur."""
        try:
            temp = int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
            raw = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True,
                                 timeout=1).stdout
            undervolt = int(raw.split("=")[-1], 16) & 1 if "=" in raw else 0
        except (OSError, ValueError, subprocess.SubprocessError):
            return 1.0
        if temp >= self._cfg["hot_c"] or undervolt:
            log.warning("motor hizi dusuruldu (isi %.1f C, dusuk voltaj %s)", temp, bool(undervolt))
            return self._cfg["derate"]
        return 1.0

    # --- disari acilan ---
    async def _cancel(self) -> None:
        # Eski gorev tamamen bitsin (finally'deki halt) ki yeni hareketin durumunu ezmesin
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def move(self, direction: str, seconds: float, speed: str = "normal",
                   duty: float | None = None, soft: bool = True) -> dict:
        """Hareketi baslatir ve hemen doner; onceki hareket once durdurulur.

        duty: hiz duzeyini atlayip dogrudan PWM (%) ver (takip, kalibrasyonla).
        soft=False: kisa darbe; rampa pulse_ramp_ms (gucu yemez, ani akimi da yumusatir).
        """
        if direction not in MOVES:
            return {"error": f"bilinmeyen yön: {direction}"}
        c = self._cfg
        if direction == "ileri" and (d := self.ahead()) is not None:
            log.info("ileri gidilmedi: onde engel %.0f cm", d)
            return {"error": "engel", "cm": d}
        # Cok kisa hareket rampada kaybolur, palet yerde kalkmaz: en az min_seconds
        seconds = max(c["min_seconds"], min(float(seconds), c["max_seconds"]))
        # Sesli donus (duty verilmez; takip ve arama kendi duty degerini verir): yerinde donmek
        # duz gitmekten fazla guc ister. Yumusak rampa kisa donusun gucunu yiyordu, kisa rampa kullan.
        spoken_turn = duty is None and direction in ("sol", "sag")
        target = min(duty, c["max_speed"]) if duty else self.duty(direction, speed)
        if spoken_turn:
            target = max(target, c.get("turn_min_speed", 0))
            soft = False
        self._derate_k = self._derate()
        duty = max(target * self._derate_k, c["min_speed"])
        await self._cancel()
        self._moving_until = time.monotonic() + seconds + 0.5
        self._task = asyncio.create_task(self._run(MOVES[direction], seconds, duty, soft), name="motor")
        log.info("hareket: %s %.1f sn %%%.0f", direction, seconds, duty)
        return {"result": f"{direction} yönünde {seconds:.1f} saniye hareket ediliyor."}

    async def _run(self, dirs: tuple[int, int], seconds: float, duty: float, soft: bool = True) -> None:
        c = self._cfg
        try:
            # Yon degisiyorsa ve az once durduysa, motor/disli icin kisa bekle
            pause = c["reverse_pause_ms"] / 1000 - (time.monotonic() - self._stopped_at)
            if self._prev_dir not in ((0, 0), dirs) and pause > 0:
                await asyncio.sleep(pause)
            self._prev_dir = dirs
            up, down = ((c["ramp_up_ms"] / 1000, c["ramp_down_ms"] / 1000) if soft
                        else (c["pulse_ramp_ms"] / 1000,) * 2)
            # Rampalar surenin en fazla %60'i; kalan surede guc hedefte kalsin
            if up + down > seconds * 0.6:
                k = seconds * 0.6 / (up + down)
                up, down = up * k, down * k
            start = min(c["ramp_start"], duty)
            heading = self.imu.heading if self.imu else None
            await self._ramp(dirs, start, duty, up)
            if heading is not None and dirs[0] == dirs[1]:
                await self._hold_heading(dirs, duty, heading, seconds - up - down)
            else:
                await self._wait(dirs, seconds - up - down)
            await self._ramp(dirs, duty, start, down)
        except Obstacle:
            d = self.sonar.distance if self.sonar else None
            self.halt()  # rampasiz, hemen
            log.info("onde engel (%s cm): durdu", "?" if d is None else f"{d:.0f}")
            if self.on_obstacle:
                self.on_obstacle(d)
        finally:
            self._steer = 0.0
            self.halt()

    @property
    def driving(self) -> bool:
        return bool(self._task and not self._task.done() and self._task.get_name() == "drive")

    async def drive(self, forward: int, duty: float, steer: float, hold_s: float) -> bool:
        """Surekli surus (kisi takibi): ileri (1) ya da geri (-1), steer ile egri cizerek.

        Durmadan guncellenir: surus suruyorsa yalniz guc/yon/bitis degisir, motor yeniden kalkmaz.
        hold_s icinde yeni komut gelmezse kendiliginden durur (takip donmazsa robot kacmasin).
        Onde engel varken ileri gitmez (False).
        """
        dirs = MOVES["ileri"] if forward > 0 else MOVES["geri"]
        if forward > 0 and self.ahead() is not None:
            if self.driving and self._drive_cmd[0] == dirs:
                await self.stop(quiet=True)
            return False
        duty = max(self._cfg["min_speed"], min(duty * self._derate_k, self._cfg["max_speed"]))
        until = time.monotonic() + hold_s
        self._drive_cmd = [dirs, duty, steer, until]
        self._moving_until = until + 0.1
        if not self.driving:
            await self._cancel()
            self._task = asyncio.create_task(self._drive_loop(), name="drive")
        return True

    async def _drive_loop(self) -> None:
        c = self._cfg
        prev = None
        try:
            while self._drive_cmd and time.monotonic() < self._drive_cmd[3]:
                dirs, duty, steer, _ = self._drive_cmd
                if prev is not None and dirs != prev:
                    # ileri <-> geri: once dur, disliler icin kisa bekle
                    self._steer = 0.0
                    self._set(0, 0, 0)
                    await asyncio.sleep(c["reverse_pause_ms"] / 1000)
                prev = self._prev_dir = dirs
                self._check(dirs)
                self._steer = steer
                self._set(*dirs, duty)
                await asyncio.sleep(0.02)
        except Obstacle:
            log.info("takipte onde engel (%s cm): durdu", "?" if self.sonar.distance is None
                     else f"{self.sonar.distance:.0f}")
        finally:
            self._drive_cmd = None
            self._steer = 0.0
            self.halt()

    async def _hold_heading(self, dirs: tuple[int, int], duty: float, heading: float, seconds: float) -> None:
        """Duz gidiste jiroskopla yonu koru: sola kaydiysa sol palet hizlanir, sag yavaslar (geride tersi)."""
        c = self._cfg
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self._check(dirs)
            drift = self.imu.heading - heading  # sola pozitif
            steer = max(-c["steer_max"], min(c["steer_max"], c["steer_per_deg"] * drift))
            self._steer = steer if dirs[0] > 0 else -steer
            self._set(*dirs, duty)
            await asyncio.sleep(min(0.02, max(0.0, end - time.monotonic())))

    async def _ramp(self, dirs: tuple[int, int], start: float, end: float, seconds: float) -> None:
        step = RAMP_STEP_S
        n = max(1, int(seconds / step))
        for i in range(1, n + 1):
            self._check(dirs)
            self._set(*dirs, start + (end - start) * i / n)
            await asyncio.sleep(step)

    async def _wait(self, dirs: tuple[int, int], seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self._check(dirs)
            await asyncio.sleep(min(0.02, max(0.0, end - time.monotonic())))

    async def signal_check(self) -> dict:
        """Takilma teshisi: kisa bir darbe ver, PWM pinlerde gercekten var mi olc.

        Sinyal varsa sorun guc tarafinda (batarya, koruma, kablo); yoksa yazilim/pin tarafinda.
        """
        side = self._sides["sol"]
        pins = (side["in1"], side["in2"])
        await self.move("sag", 0.5, duty=self._cfg["min_speed"] + 4, soft=False)
        await asyncio.sleep(0.15)
        highs = 0
        for _ in range(40):
            out = (await asyncio.create_subprocess_exec(
                "pinctrl", "get", ",".join(map(str, pins)), stdout=asyncio.subprocess.PIPE)).stdout
            highs += (await out.read()).count(b"| hi")
            await asyncio.sleep(0.004)
        await self.stop()
        ok = highs > 0
        log.warning("motor sinyal kontrolu: PWM %s (40 olcumde %d yuksek)", "VAR" if ok else "YOK", highs)
        return {"pwm_sinyali": "var" if ok else "yok"}

    async def stop(self, quiet: bool = False) -> dict:
        """quiet: takibin erken durdurmasi (hedef ortaya geldi), her seferinde loglanmaz."""
        await self._cancel()
        self.halt()
        if not quiet:
            log.info("motorlar durduruldu")
        return {"result": "Durdum."}

    def close(self) -> None:
        self.halt()
        self._lg.gpiochip_close(self._h)
