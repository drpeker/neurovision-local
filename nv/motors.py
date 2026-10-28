"""Palet motorlari (MX1508 ya da L298N). Guvenlik kurallari burada, modelin kararina birakilmaz.

MX1508'de EN yok: hiz, yone gore IN1 ya da IN2'ye verilen PWM ile ayarlanir, digeri 0.
L298N'de yon IN'lerde, PWM EN'de. Tum hizlar hiz duzeyiyle (%, sesle ayarlanir) carpilir.

- her hareket en fazla max_seconds; yeni komut eskisini iptal eder
- yon degisirken once dur, kisa bekle
- yumusak kalkis (ramp_start'tan hedefe ramp_up_ms) ve yumusak durus (ramp_down_ms);
  stop komutu, baglanti kopmasi ve cikista ise ANINDA durur
- sicak ya da dusuk voltajda hiz dusurulur
- motor calisirken (ve durduktan sonra mute_tail_ms) mikrofon susar: motor sesi modele gitmez
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
        self._task: asyncio.Task | None = None
        self._prev_dir: tuple[int, int] = (0, 0)  # son hareketin yonu (durunca sifirlanmaz)
        self._stopped_at = 0.0
        self._moving_until = 0.0
        self._quiet_until = 0.0
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
        for name, direction in (("sol", left), ("sag", right)):
            side = self._sides[name]
            self._drive(side, direction, duty * side.get("trim", 1.0))

    def halt(self) -> None:
        """Senkron, aninda dur (cikis, hata, baglanti kopmasi)."""
        self._set(0, 0, 0)
        self._stopped_at = time.monotonic()
        self._moving_until = 0.0
        self._quiet_until = self._stopped_at + self._cfg["mute_tail_ms"] / 1000

    # --- hiz siniri ---
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
        # Cok kisa hareket rampada kaybolur, palet yerde kalkmaz: en az min_seconds
        seconds = max(c["min_seconds"], min(float(seconds), c["max_seconds"]))
        # Sesli donus (duty verilmez; takip ve arama kendi duty degerini verir): yerinde donmek
        # duz gitmekten fazla guc ister. Yumusak rampa kisa donusun gucunu yiyordu, kisa rampa kullan.
        spoken_turn = duty is None and direction in ("sol", "sag")
        target = min(duty, c["max_speed"]) if duty else self.duty(direction, speed)
        if spoken_turn:
            target = max(target, c.get("turn_min_speed", 0))
            soft = False
        duty = max(target * self._derate(), c["min_speed"])
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
            await self._ramp(dirs, start, duty, up)
            await asyncio.sleep(seconds - up - down)
            await self._ramp(dirs, duty, start, down)
        finally:
            self.halt()

    async def _ramp(self, dirs: tuple[int, int], start: float, end: float, seconds: float) -> None:
        step = RAMP_STEP_S
        n = max(1, int(seconds / step))
        for i in range(1, n + 1):
            self._set(*dirs, start + (end - start) * i / n)
            await asyncio.sleep(step)

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

    async def stop(self) -> dict:
        await self._cancel()
        self.halt()
        log.info("motorlar durduruldu")
        return {"result": "Durdum."}

    def close(self) -> None:
        self.halt()
        self._lg.gpiochip_close(self._h)
