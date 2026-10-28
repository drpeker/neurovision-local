"""Wi-Fi kurulumu: kayitli aga baglanamazsa robot kendi agini acar, kurulum sayfasi sunar.

Robot programindan bagimsiz sistem servisi (root: ag ayarlari). Akis:
  1. Acilista WAIT_S kadar kayitli aga baglanmasi beklenir; baglanirsa cikilir.
  2. Baglanamazsa cevredeki aglar taranir, "NeuroVision-Kurulum" agi acilir.
  3. Telefon bu aga baglaninca kurulum sayfasi kendiliginden acilir (her DNS sorgusu
     10.42.0.1'e gider, NetworkManager dnsmasq ayari install-wifi.sh'te). Acilmazsa: 10.42.0.1
  4. Secilen aga baglanilir; basarirsa NetworkManager agi kaydeder (sonraki acilista otomatik),
     basaramazsa kurulum agi yeniden acilir ve sayfada hata gosterilir.
  5. Kurulum agi acikken RETRY_S'de bir, sayfa kullanilmiyorsa kayitli aglar yeniden denenir.

Ekran: robot programi STATUS dosyasini okuyup altyazi olarak gosterir.
Yalnizca standart kutuphane; nmcli ile calisir.
"""
import html
import logging
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

IFACE = "wlan0"
AP_SSID = "NeuroVision-Kurulum"
AP_PASSWORD = "neurovision"  # en az 8 karakter (WPA2)
AP_CON = "nv-kurulum"
AP_IP = "10.42.0.1"
WAIT_S = 30
RETRY_S = 300
IDLE_S = 120  # sayfa bu kadar suredir kullanilmiyorsa kayitli aglar denenebilir
STATUS = Path("/run/neurovision-wifi.txt")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("wifi")


def nmcli(*args: str, timeout: float = 30) -> subprocess.CompletedProcess:
    return subprocess.run(["nmcli", *args], capture_output=True, text=True, timeout=timeout)


def split_terse(line: str) -> list[str]:
    """nmcli -t satiri: alanlar ':' ile ayrilir, alan icindeki ':' '\\:' olarak kacirilir."""
    out, cur, esc = [], "", False
    for ch in line:
        if esc:
            cur, esc = cur + ch, False
        elif ch == "\\":
            esc = True
        elif ch == ":":
            out, cur = out + [cur], ""
        else:
            cur += ch
    return out + [cur]


def connected() -> bool:
    """wlan0 kurulum agi disinda bir aga bagli mi?"""
    r = nmcli("-t", "-f", "GENERAL.STATE,GENERAL.CONNECTION", "device", "show", IFACE)
    info = dict(split_terse(l)[:2] for l in r.stdout.splitlines() if ":" in l)
    return info.get("GENERAL.STATE", "").startswith("100") and info.get("GENERAL.CONNECTION") != AP_CON


def wait_connected(seconds: float) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if connected():
            return True
        time.sleep(2)
    return connected()


def scan() -> list[dict]:
    """Cevredeki aglar, sinyale gore; ayni ad bir kez (en guclu)."""
    nmcli("device", "wifi", "rescan", "ifname", IFACE)
    time.sleep(4)
    r = nmcli("-t", "-f", "SSID,SIGNAL,SECURITY", "device", "wifi", "list", "ifname", IFACE)
    best: dict[str, dict] = {}
    for line in r.stdout.splitlines():
        f = split_terse(line)
        if len(f) < 3 or not f[0] or f[0] == AP_SSID:
            continue
        net = {"ssid": f[0], "signal": int(f[1] or 0), "secure": f[2] not in ("", "--")}
        if net["ssid"] not in best or net["signal"] > best[net["ssid"]]["signal"]:
            best[net["ssid"]] = net
    return sorted(best.values(), key=lambda n: -n["signal"])


def status(text: str | None) -> None:
    if text:
        STATUS.write_text(text)
    else:
        STATUS.unlink(missing_ok=True)


def ap_up() -> None:
    nmcli("connection", "delete", AP_CON)
    r = nmcli("connection", "add", "type", "wifi", "ifname", IFACE, "con-name", AP_CON,
              "autoconnect", "no", "ssid", AP_SSID,
              "802-11-wireless.mode", "ap", "802-11-wireless.band", "bg",
              "ipv4.method", "shared", "ipv4.addresses", f"{AP_IP}/24", "ipv6.method", "disabled",
              "wifi-sec.key-mgmt", "wpa-psk", "wifi-sec.psk", AP_PASSWORD)
    if r.returncode:
        log.error("kurulum agi tanimlanamadi: %s", r.stderr.strip())
    r = nmcli("connection", "up", AP_CON)
    if r.returncode:
        log.error("kurulum agi acilamadi: %s", r.stderr.strip())
    log.info("kurulum agi acik: %s", AP_SSID)
    status(f"Wi-Fi yok. Kurulum: '{AP_SSID}' ağına bağlan, şifre {AP_PASSWORD}, sonra {AP_IP}")


def ap_down() -> None:
    nmcli("connection", "down", AP_CON)
    nmcli("connection", "delete", AP_CON)


def try_known() -> bool:
    """Kurulum agini kapatip kayitli aglara bir sans ver."""
    ap_down()
    nmcli("device", "connect", IFACE)
    return wait_connected(WAIT_S)


def connect(ssid: str, password: str) -> str | None:
    """Basariliysa None, degilse hata metni. Basarisiz denemenin profili silinir."""
    ap_down()
    nmcli("device", "wifi", "rescan", "ifname", IFACE)
    time.sleep(4)
    args = ["device", "wifi", "connect", ssid, "ifname", IFACE]
    if password:
        args += ["password", password]
    try:
        r = nmcli(*args, timeout=60)
    except subprocess.TimeoutExpired:
        r = None
    if r and r.returncode == 0 and wait_connected(15):
        log.info("baglanildi: %s", ssid)
        return None
    err = r.stderr.strip() if r else "zaman aşımı"
    log.warning("baglanilamadi: %s (%s)", ssid, err)
    nmcli("connection", "delete", ssid)  # yanlis sifreyle kayitli kalmasin
    if "Secrets were required" in err or "802-1x" in err or "psk" in err.lower():
        return "Şifre yanlış olabilir."
    if "No network with SSID" in err:
        return "Ağ bulunamadı."
    return "Bağlanılamadı."


PAGE = """<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>NeuroVision Wi-Fi</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;padding:16px;background:#0d1117;color:#e6edf3}}
h1{{font-size:20px;margin:0 0 4px}} p{{color:#9da7b3;margin:0 0 16px}}
label.net{{display:flex;justify-content:space-between;padding:12px;border:1px solid #30363d;
border-radius:8px;margin-bottom:8px;background:#161b22}}
label.net input{{margin-right:10px}} .sig{{color:#9da7b3;white-space:nowrap}}
input[type=password],input[type=text]{{width:100%;box-sizing:border-box;padding:12px;font-size:16px;
border-radius:8px;border:1px solid #30363d;background:#0d1117;color:#e6edf3;margin:8px 0}}
button{{width:100%;padding:14px;font-size:16px;border:0;border-radius:8px;background:#28a8ff;color:#fff}}
a.btn{{display:block;text-align:center;margin-top:12px;color:#28a8ff}}
.err{{background:#3d1214;color:#ff9b9b;padding:10px;border-radius:8px;margin-bottom:12px}}
</style></head><body>
<h1>NeuroVision Wi-Fi kurulumu</h1><p>Robotun bağlanacağı ağı seç ve şifresini yaz.</p>
{error}
<form method="post" action="/baglan">
{nets}
<label class="net"><span><input type="radio" name="ssid" value="__diger__">Listede yok (gizli ağ)</span></label>
<input type="text" name="other" placeholder="Gizli ağın adı" autocomplete="off">
<input type="password" name="password" placeholder="Şifre (açık ağda boş bırak)" autocomplete="off">
<button type="submit">Bağlan</button></form>
<a class="btn" href="/tara">Yeniden tara (bağlantı birkaç saniye kopar)</a>
</body></html>"""

DONE = """<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>NeuroVision Wi-Fi</title>
<style>body{{font-family:system-ui,sans-serif;padding:16px;background:#0d1117;color:#e6edf3}}</style>
</head><body><h1>Bağlanılıyor: {ssid}</h1>
<p>Robot kurulum ağını kapatıp bu ağa geçiyor; telefonun kurulum ağından kopacak.</p>
<p>Bağlanamazsa kurulum ağı yaklaşık bir dakika içinde yeniden açılır; o zaman bu sayfaya
yeniden gir, hata mesajını göreceksin.</p></body></html>"""


class Portal:
    def __init__(self) -> None:
        self.nets: list[dict] = []
        self.error = ""
        self.busy = False  # baglanma/tarama suruyor
        self.last_request = 0.0
        self.done = threading.Event()

    def page(self) -> str:
        rows = []
        for i, n in enumerate(self.nets):
            bars = "▂▄▆█"[: max(1, min(4, n["signal"] // 25 + 1))]
            lock = " 🔒" if n["secure"] else ""
            checked = " checked" if i == 0 else ""
            rows.append(f'<label class="net"><span><input type="radio" name="ssid" '
                        f'value="{html.escape(n["ssid"])}"{checked}>{html.escape(n["ssid"])}{lock}</span>'
                        f'<span class="sig">{bars} {n["signal"]}%</span></label>')
        error = f'<div class="err">{html.escape(self.error)}</div>' if self.error else ""
        return PAGE.format(nets="\n".join(rows) or "<p>Ağ bulunamadı; yeniden tara.</p>", error=error)

    def rescan(self) -> None:
        self.busy = True
        try:
            ap_down()
            self.nets = scan()
            log.info("tarama: %d ag", len(self.nets))
        finally:
            ap_up()
            self.busy = False

    def join(self, ssid: str, password: str) -> None:
        self.busy = True
        status(f"Wi-Fi: {ssid} ağına bağlanılıyor...")
        err = connect(ssid, password)
        if err is None:
            self.error = ""
            status(None)
            self.done.set()
        else:
            self.error = f"{ssid}: {err}"
            self.nets = scan()
            ap_up()
        self.busy = False


def handler(portal: Portal):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def send(self, body: str, code: int = 200) -> None:
            data = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def redirect(self, to: str = f"http://{AP_IP}/") -> None:
            self.send_response(302)
            self.send_header("Location", to)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            portal.last_request = time.monotonic()
            host = (self.headers.get("Host") or "").split(":")[0]
            # Telefonlarin "internet var mi" kontrolleri (generate_204, hotspot-detect...) ve
            # baska her adres kurulum sayfasina yonlenir: sayfa kendiliginden acilir
            if host != AP_IP:
                return self.redirect()
            if self.path.startswith("/tara"):
                threading.Thread(target=portal.rescan, daemon=True).start()
                return self.send('<meta http-equiv="refresh" content="12;url=/">'
                                 "<p style='font-family:sans-serif'>Taranıyor; birkaç saniye sonra "
                                 "kurulum ağına yeniden bağlanıp sayfayı yenile.</p>")
            self.send(portal.page())

        def do_POST(self):
            portal.last_request = time.monotonic()
            n = int(self.headers.get("Content-Length") or 0)
            form = urllib.parse.parse_qs(self.rfile.read(min(n, 4096)).decode(), keep_blank_values=True)
            ssid = form.get("ssid", [""])[0]
            if ssid == "__diger__":
                ssid = form.get("other", [""])[0].strip()
            password = form.get("password", [""])[0]
            if not ssid or portal.busy:
                return self.redirect()
            if password and not 8 <= len(password) <= 63:
                portal.error = "WPA şifresi 8-63 karakter olmalı."
                return self.redirect()
            self.send(DONE.format(ssid=html.escape(ssid)))
            # Cevap telefona gitsin, sonra kurulum agi kapansin
            threading.Timer(1.5, portal.join, args=(ssid, password)).start()
    return H


def main() -> None:
    status(None)
    # --zorla: bagliyken de kurulumu ac (test icin; SSH baglantisi kopar)
    if "--zorla" not in sys.argv and wait_connected(WAIT_S):
        log.info("kayitli aga bagli, kurulum gerekmiyor")
        return
    log.info("kayitli aga baglanilamadi, kurulum basliyor")
    portal = Portal()
    portal.nets = scan()
    ap_up()
    server = ThreadingHTTPServer(("0.0.0.0", 80), handler(portal))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    next_retry = time.monotonic() + RETRY_S
    try:
        while not portal.done.wait(5):
            idle = time.monotonic() - portal.last_request > IDLE_S
            if time.monotonic() > next_retry and idle and not portal.busy:
                log.info("kayitli aglar yeniden deneniyor")
                if try_known():
                    break
                portal.nets = scan()
                ap_up()
                next_retry = time.monotonic() + RETRY_S
    finally:
        server.shutdown()
        if not connected():
            ap_down()
        status(None)
    log.info("kurulum bitti")


if __name__ == "__main__":
    main()
