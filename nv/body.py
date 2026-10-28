"""Oz farkindalik: robotun kendi donanim durumu. Yalnizca okur; hicbir pine yazmaz.

Kaynaklar sudo gerektirmez: pinctrl, vcgencmd, /proc, /sys, /dev/i2c-N (i2c grubu).
Modele giden metinler Turkce; ayrintili veri ve kisa ozet birlikte doner.
"""

import asyncio
import errno
import fcntl
import os
import re
import shutil
import time
from pathlib import Path

from .camera import Camera
from .memory import Memory
from .vad import SpeechGate

I2C_SLAVE = 0x0703
# Sik kullanilan I2C cipleri (adres -> olasi ad)
KNOWN_I2C = {
    0x1E: "HMC5883L pusula", 0x23: "BH1750 ışık sensörü", 0x29: "VL53L0X mesafe / TSL2591 ışık",
    0x38: "AHT20 sıcaklık-nem", 0x3C: "SSD1306 OLED ekran", 0x40: "INA219 akım-voltaj / PCA9685 PWM",
    0x41: "INA219 akım-voltaj", 0x44: "SHT3x sıcaklık-nem", 0x48: "ADS1115 ADC", 0x53: "ADXL345 ivmeölçer",
    0x57: "EEPROM", 0x5A: "MLX90614 kızılötesi termometre", 0x68: "MPU6050 IMU / DS3231 saat",
    0x69: "MPU6050 IMU", 0x70: "PCA9685 (genel adres) / TCA9548A çoklayıcı", 0x76: "BME280 / BMP280",
    0x77: "BME280 / BMP280 / BMP180",
}
THROTTLE_BITS = {
    0: "düşük voltaj", 1: "frekans sınırlı", 2: "kısıtlanıyor", 3: "yumuşak ısı sınırı",
    16: "düşük voltaj", 17: "frekans sınırlandı", 18: "kısıtlandı", 19: "yumuşak ısı sınırı",
}
SECTIONS = ["ozet", "guc", "isi", "cpu", "ram", "disk", "ag", "gpio", "i2c", "isik", "ses",
            "motor", "gorus", "sistemler", "kota", "hepsi"]
USAGE_LABELS = {"in_TEXT": "giriş_metin", "in_AUDIO": "giriş_ses", "in_IMAGE": "giriş_görüntü",
                "in_VIDEO": "giriş_video", "out_AUDIO": "çıkış_ses", "out_TEXT": "çıkış_metin"}
SOURCE_LABELS = {"live": "sohbet (Gemini Live)", "ozet": "hafıza özeti (Flash-Lite)"}


def _read(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


async def _run(*cmd: str) -> str:
    try:
        p = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(p.communicate(), timeout=3)
        return out.decode(errors="replace").strip()
    except (OSError, asyncio.TimeoutError):
        return ""


def _cpu_times() -> list[tuple[int, int]]:
    rows = []
    for line in _read("/proc/stat").splitlines():
        if line.startswith("cpu"):
            v = [int(x) for x in line.split()[1:]]
            idle = v[3] + v[4]
            rows.append((sum(v) - idle, sum(v)))
    return rows


def _scan_i2c(bus: int) -> list[dict]:
    """i2cdetect -r gibi: her adreste 1 bayt okumayi dene. Surucunun tuttugu adres 'mesgul'."""
    found = []
    try:
        fd = os.open(f"/dev/i2c-{bus}", os.O_RDWR)
    except OSError as e:
        return [{"hata": str(e)}]
    try:
        for addr in range(0x08, 0x78):
            try:
                fcntl.ioctl(fd, I2C_SLAVE, addr)
                os.read(fd, 1)
                found.append({"adres": f"0x{addr:02X}", "olası": KNOWN_I2C.get(addr, "bilinmiyor")})
            except OSError as e:
                if e.errno == errno.EBUSY:
                    found.append({"adres": f"0x{addr:02X}", "olası": KNOWN_I2C.get(addr, "bilinmiyor"),
                                  "durum": "çekirdek sürücüsü kullanıyor"})
    finally:
        os.close(fd)
    return found


def _light_word(lux: float) -> str:
    for limit, word in ((5, "karanlık"), (50, "loş"), (300, "normal iç mekân"),
                        (1000, "aydınlık"), (float("inf"), "çok aydınlık / gün ışığı")):
        if lux < limit:
            return word
    return ""


class Body:
    def __init__(self, cfg: dict, camera: Camera | None, gate: SpeechGate, memory: Memory):
        self._cfg = cfg["body"]
        self._camera = camera
        self._gate = gate
        self._memory = memory
        self._margin = cfg["vad"]["margin_db"]
        self._started = time.monotonic()
        self.connected_since: float | None = None  # LiveBrain doldurur
        self.speaker = None  # LiveBrain doldurur
        self.radio = None  # LiveBrain.attach_radio
        self.motors = None  # LiveBrain.attach_motors
        self.tracker = None  # robot.py
        self._pin_names = {int(k): v for k, v in self._cfg["pin_names"].items()}
        self._reserved = {int(k): v for k, v in self._cfg["reserved_pins"].items()}

    # --- bolumler ---
    async def _power(self) -> dict:
        raw = await _run("vcgencmd", "get_throttled")
        bits = int(raw.split("=")[-1], 16) if "=" in raw else 0
        volts = await _run("vcgencmd", "measure_volts", "core")
        return {
            "şu_an": [THROTTLE_BITS[b] for b in range(4) if bits >> b & 1] or ["normal"],
            "açılıştan_beri": [THROTTLE_BITS[b] for b in range(16, 20) if bits >> b & 1] or ["sorun yok"],
            "çekirdek_voltajı": volts.split("=")[-1] if volts else "?",
            "not": "Pi 4 5 V girişi doğrudan ölçemez; düşük voltaj bayrağı ~4.63 V altını gösterir.",
        }

    def _temp(self) -> dict:
        t = _read("/sys/class/thermal/thermal_zone0/temp")
        return {"işlemci_°C": round(int(t) / 1000, 1) if t else None}

    async def _cpu(self) -> dict:
        a = _cpu_times()
        await asyncio.sleep(0.3)
        b = _cpu_times()
        use = [round(100 * (b[i][0] - a[i][0]) / max(1, b[i][1] - a[i][1])) for i in range(len(a))]
        freq = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
        return {"toplam_%": use[0], "çekirdekler_%": use[1:],
                "yük_ortalaması": _read("/proc/loadavg").split()[:3],
                "frekans_MHz": int(freq) // 1000 if freq else None}

    def _ram(self) -> dict:
        info = {}
        for line in _read("/proc/meminfo").splitlines():
            k, _, v = line.partition(":")
            info[k] = int(v.split()[0]) // 1024
        rss = re.search(r"VmRSS:\s+(\d+)", _read("/proc/self/status"))
        return {"toplam_MB": info.get("MemTotal"), "boş_MB": info.get("MemAvailable"),
                "kullanım_%": round(100 * (1 - info.get("MemAvailable", 0) / max(1, info.get("MemTotal", 1)))),
                "swap_kullanılan_MB": info.get("SwapTotal", 0) - info.get("SwapFree", 0),
                "robot_süreci_MB": int(rss.group(1)) // 1024 if rss else None}

    def _disk(self) -> dict:
        d = shutil.disk_usage("/")
        return {"boş_GB": round(d.free / 1e9, 1), "kullanım_%": round(100 * d.used / d.total)}

    async def _net(self) -> dict:
        level = None
        for line in _read("/proc/net/wireless").splitlines()[2:]:
            parts = line.split()
            if parts and parts[0].rstrip(":") == "wlan0":
                level = int(float(parts[3]))
        ip = (await _run("ip", "-4", "-br", "addr", "show", "wlan0")).split()
        return {
            "wifi_sinyal_dBm": level,
            "wifi_kalitesi": None if level is None else
            ("iyi" if level > -60 else "orta" if level > -72 else "zayıf"),
            "ip": ip[2].split("/")[0] if len(ip) > 2 else None,
            "gemini_bağlantısı": "bağlı" if self.connected_since else "bağlı değil",
            "bağlantı_süresi_dk": round((time.monotonic() - self.connected_since) / 60, 1)
            if self.connected_since else None,
        }

    async def _gpio(self) -> dict:
        pins = {}
        for line in (await _run("pinctrl", "get", "0-27")).splitlines():
            # Cikis pinlerinde surus alani da var: "5: op -- pu | hi // GPIO5 = output"
            m = re.match(r"\s*(\d+):\s+(\S+)(?:\s+(?:--|dh|dl))?\s+(\S+)\s+\|\s+(\S+)\s+//\s+\S+\s+=\s+(\S+)",
                         line)
            if not m:
                continue
            n, fn, pull, level, name = int(m[1]), m[2], m[3], m[4], m[5]
            if n in self._reserved:
                pins[f"GPIO{n}"] = f"sistem: {self._reserved[n]} ({name})"
                continue
            mode = {"ip": "giriş", "op": "çıkış"}.get(fn, f"alternatif ({name})")
            pull_word = {"pu": "pull-up", "pd": "pull-down", "pn": "pull yok"}.get(pull, pull)
            label = f" [{self._pin_names[n]}]" if n in self._pin_names else ""
            pins[f"GPIO{n}{label}"] = f"{mode}, {'yüksek' if level == 'hi' else 'düşük'}, {pull_word}"
        return pins or {"hata": "pinctrl okunamadı"}

    async def _i2c(self) -> dict:
        loop = asyncio.get_running_loop()
        return {f"i2c-{b}": await loop.run_in_executor(None, _scan_i2c, b)
                for b in self._cfg["i2c_buses"]}

    def _light(self) -> dict:
        if not self._camera or not self._camera.meta:
            return {"durum": "kamera ölçümü yok"}
        m = self._camera.meta
        lux = m.get("lux", 0.0)
        return {"lux_tahmini": round(lux, 1), "yorum": _light_word(lux),
                "renk_sıcaklığı_K": int(m.get("colour_k", 0)) or None,
                "kaynak": "kameranın pozlama/kazanç tahmini (ışık sensörü değil)",
                "ölçüm_yaşı_sn": round(time.monotonic() - self._camera.meta_at, 1)}

    def _sound(self) -> dict:
        out = {"ortam_tabanı_dBFS": round(self._gate.floor_db, 1),
               "konuşma_eşiği_dBFS": round(self._gate.floor_db + self._margin, 1)}
        if self.speaker:
            out["hoparlör_ses_düzeyi_%"] = self.speaker.volume
        if self.radio:
            out["radyo"] = self.radio.playing or "kapalı"
        return out

    def _motor(self) -> dict:
        return self.motors.status() if self.motors else {"durum": "motorlar bağlı değil"}

    def _vision(self) -> dict:
        return self.tracker.status() if self.tracker else {"durum": "nesne tanıma kapalı"}

    def _systems(self) -> dict:
        cam_age = time.monotonic() - self._camera.latest_at if self._camera and self._camera.latest else None
        return {"kamera": "yok" if cam_age is None else
                ("çalışıyor" if cam_age < 3 else f"son kare {int(cam_age)} sn önce"),
                "sistem_açık_saat": round(float(_read("/proc/uptime").split()[0]) / 3600, 1),
                "robot_açık_dk": round((time.monotonic() - self._started) / 60, 1),
                "hafıza": self._memory.count()}

    def _usage(self) -> dict:
        today: dict[str, dict] = {}
        for source, kind, tokens, calls in self._memory.usage():
            s = today.setdefault(SOURCE_LABELS.get(source, source), {})
            if kind == "total":
                s["toplam_token"], s["istek_sayısı"] = tokens, calls
            else:
                s[USAGE_LABELS.get(kind, kind)] = tokens
        return {
            "bugün": today or "henüz kullanım yok",
            "bugün_toplam_token": sum(s.get("toplam_token", 0) for s in today.values()),
            "son_7_gün": {day: {"token": t, "istek": c} for day, t, c in self._memory.usage_days(7)},
            "not": "Sohbette her tur, konuşmanın o ana kadarki tamamını yeniden işler; giriş tokenı "
                   "bu yüzden konuşma uzadıkça artar. Kota sınırları proje bazında, AI Studio'da görünür.",
        }

    # --- disari acilan ---
    async def status(self, section: str = "ozet") -> dict:
        section = section if section in SECTIONS else "ozet"
        parts = {
            "guc": self._power, "isi": self._temp, "cpu": self._cpu, "ram": self._ram,
            "disk": self._disk, "ag": self._net, "gpio": self._gpio, "i2c": self._i2c,
            "isik": self._light, "ses": self._sound, "motor": self._motor, "gorus": self._vision,
            "sistemler": self._systems,
            "kota": self._usage,
        }
        if section == "ozet":
            wanted = ["guc", "isi", "cpu", "ram", "disk", "ag", "isik", "ses", "motor", "sistemler"]
        elif section == "hepsi":
            wanted = list(parts)
        else:
            wanted = [section]
        out = {}
        for key in wanted:
            r = parts[key]()
            out[key] = await r if asyncio.iscoroutine(r) else r
        if section == "ozet":
            out["bugünkü_gemini_token"] = self._usage()["bugün_toplam_token"]
        out["uyarılar"] = self._warnings(out)
        return out

    def _warnings(self, s: dict) -> list[str]:
        w = []
        t = s.get("isi", {}).get("işlemci_°C")
        if t and t >= self._cfg["warn_temp_c"]:
            w.append(f"işlemci sıcak: {t} °C")
        power = s.get("guc", {})
        if "düşük voltaj" in power.get("şu_an", []):
            w.append("şu an düşük voltaj var")
        elif "düşük voltaj" in power.get("açılıştan_beri", []):
            w.append("açılıştan beri en az bir kez düşük voltaj oldu")
        ram = s.get("ram", {})
        if ram.get("kullanım_%", 0) >= 85:
            w.append(f"bellek dolu: %{ram['kullanım_%']}")
        if s.get("disk", {}).get("kullanım_%", 0) >= 90:
            w.append("disk dolmak üzere")
        net = s.get("ag", {})
        if net.get("wifi_kalitesi") == "zayıf":
            w.append(f"wifi zayıf: {net['wifi_sinyal_dBm']} dBm")
        cam = s.get("sistemler", {}).get("kamera", "")
        if cam.startswith("son kare"):
            w.append(f"kamera görüntü vermiyor ({cam})")
        return w
