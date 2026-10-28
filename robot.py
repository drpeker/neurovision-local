"""NeuroVision robot: Gemini Live ile sesli ve gorsel sohbet.

    .venv/bin/python robot.py            # robotu calistir
    .venv/bin/python robot.py --check    # donanim testi (API anahtari gerekmez)
"""

import argparse
import asyncio
import logging
import logging.handlers
import os
import signal
import subprocess
import sys
import tomllib
from pathlib import Path

from nv.audio import Mic, Speaker, set_mic_boost
from nv.camera import Camera
from nv.radio import Radio
from nv.vad import SpeechGate

ROOT = Path(__file__).resolve().parent
log = logging.getLogger("robot")


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def setup_logging(cfg: dict) -> None:
    path = ROOT / cfg["file"]
    path.parent.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    file_h = logging.handlers.RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3)
    console_h = logging.StreamHandler(sys.stdout)
    for h in (file_h, console_h):
        h.setFormatter(fmt)
    logging.basicConfig(level=cfg["level"], handlers=[file_h, console_h])
    for noisy in ("websockets", "httpx", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def make_camera(cfg: dict) -> Camera | None:
    c = cfg["camera"]
    if not c["enabled"]:
        return None
    return Camera(c["width"], c["height"], c["fps"], c["rotation"], c.get("hflip", False))


async def check(cfg: dict) -> None:
    """3 sn kaydet, kac karede konusma oldugunu say, hoparlorden geri cal, kamera karesi al."""
    a, v = cfg["audio"], cfg["vad"]
    mic, speaker = Mic(a["mic"], a["highpass_hz"]), Speaker(a["speaker"], a["echo_tail_ms"])
    gate = SpeechGate(v["aggressiveness"], v["preroll_ms"], v["hangover_ms"])
    camera = make_camera(cfg)
    if camera:
        await camera.start()
    mic.start()
    print("3 saniye konus...")
    frames, voiced = [], 0
    while len(frames) < 150:
        f = await mic.frames.get()
        frames.append(f)
        voiced += gate._vad.is_speech(f, 16000)
    mic.stop()
    peak = max(abs(int.from_bytes(f[i:i + 2], "little", signed=True))
               for f in frames for i in range(0, len(f), 2))
    print(f"mikrofon: {len(frames)} kare, konusma {voiced} kare, tepe {peak / 32768:.2f}")

    # 16 kHz -> 24 kHz: her 2 ornegi 3'e cikar (sadece test icin kaba donusum)
    pcm16 = b"".join(frames)
    out = bytearray()
    for i in range(0, len(pcm16) - 3, 4):
        s0, s1 = pcm16[i:i + 2], pcm16[i + 2:i + 4]
        out += s0 + s0 + s1
    speaker.start()
    print("kayit geri calaniyor...")
    speaker.play(bytes(out))
    while speaker.busy:
        await asyncio.sleep(0.1)
    speaker.stop()

    if camera:
        await asyncio.sleep(1.5)
        jpeg = camera.frame()
        if jpeg:
            (ROOT / "logs").mkdir(exist_ok=True)
            (ROOT / "logs/check.jpg").write_bytes(jpeg)
            print(f"kamera: {len(jpeg)} bayt JPEG -> logs/check.jpg")
        else:
            print("kamera: kare alinamadi")
        await camera.stop()


async def run(cfg: dict) -> None:
    from nv.live import LiveBrain
    from nv.memory import Memory

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit(f"GEMINI_API_KEY yok: {ROOT / '.env'} dosyasina ekle (.env.example'a bak)")

    memory = Memory(ROOT / cfg["memory"]["db"])
    log.info("hafiza: %s", memory.count())
    a, r = cfg["audio"], cfg["radio"]
    # Komutla degistirilen ses duzeyi hafizada saklanir, yeniden baslayinca korunur
    volume = int(memory.get("volume") or a["volume"])
    mic = Mic(a["mic"], a["highpass_hz"])
    speaker = Speaker(a["speaker"], a["echo_tail_ms"], volume, r["level"], r["duck"])
    camera = make_camera(cfg)
    if camera:
        await camera.start()
    speaker.start()
    mic.start()
    brain = LiveBrain(cfg, key, mic, speaker, camera, memory)
    radio = Radio(speaker, r["stations"]) if r["enabled"] else None
    if radio:
        brain.attach_radio(radio)
    face = await make_face(cfg, speaker, brain)
    if face:
        face.radio = radio
    motors = make_motors(cfg, brain, memory)
    tracker = make_tracker(cfg, camera, motors, brain, face)

    task = asyncio.create_task(brain.run(), name="brain")
    stop_target = []

    def on_shutdown(target: str) -> None:
        stop_target.append(target)
        task.cancel()

    brain.attach_shutdown(on_shutdown)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, task.cancel)
    log.info("NeuroVision hazir. Durdurmak icin Ctrl+C.")
    try:
        await task
    except asyncio.CancelledError:
        pass
    finally:
        if tracker:
            await tracker.stop()
        if motors:
            motors.close()  # once motorlar: her durumda dur
        if radio:
            await radio.stop()
        mic.stop()
        speaker.stop()
        if camera:
            await camera.stop()
        if face:
            await face.stop()
        memory.close()
        log.info("kapatildi")
    if stop_target == ["sistem"]:
        # sudoers izni gerekir: deploy/allow-poweroff.sh
        r = subprocess.run(["sudo", "-n", "systemctl", "poweroff"], capture_output=True, text=True)
        if r.returncode:
            log.warning("Pi kapatilamadi (deploy/allow-poweroff.sh calistirildi mi?): %s", r.stderr.strip())


def make_tracker(cfg: dict, camera, motors, brain, face):
    """Nesne takibi: kamera, motor ve model dosyasi varsa."""
    v = cfg["vision"]
    model = ROOT / v["model"]
    if not (v["enabled"] and camera and motors):
        return None
    if not model.exists():
        log.warning("nesne takibi kapali: model yok (%s); deploy/get-model.sh", model)
        return None
    from nv.tracker import Tracker
    from nv.vision import Detector
    tracker = Tracker(v, camera, motors, Detector(model, v["threads"], v["threshold"]))
    tracker.face = face
    brain.attach_tracker(tracker)
    return tracker


def make_motors(cfg: dict, brain, memory):
    """Motor surucusu acilmazsa robot hareketsiz calismaya devam eder."""
    if not cfg["motors"]["enabled"]:
        return None
    from nv.motors import Motors
    try:
        motors = Motors(cfg["motors"])
    except Exception as e:
        log.warning("motorlar acilamadi, hareketsiz devam: %s", e)
        return None
    # Sesle degistirilen hiz duzeyi hafizada saklanir
    saved = memory.get("motor_speed")
    if saved:
        motors.speed_level = int(saved)
    brain.attach_motors(motors)
    return motors


async def make_face(cfg: dict, speaker: Speaker, brain):
    """Ekran yoksa ya da acilmazsa robot ekransiz calismaya devam eder."""
    if not cfg["display"]["enabled"]:
        return None
    from nv.display import Face, Panel
    try:
        panel = Panel(cfg["display"])
    except Exception as e:
        log.warning("ekran acilamadi, ekransiz devam: %s", e)
        return None
    face = Face(cfg["display"], panel, speaker, brain.gate, brain.body)
    await face.start()
    brain.attach_face(face)
    asyncio.create_task(show_wifi_setup(face), name="wifi-durum")
    return face


async def show_wifi_setup(face) -> None:
    """Wi-Fi kurulum servisi (deploy/wifi) calisirken talimati altyazida tut."""
    status = Path("/run/neurovision-wifi.txt")
    while True:
        try:
            text = status.read_text().strip()
        except OSError:
            text = ""
        if text:
            face.caption(text, "warn")
        await asyncio.sleep(5)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="donanim testi")
    p.add_argument("--config", default=str(ROOT / "config.toml"))
    args = p.parse_args()

    cfg = tomllib.loads(Path(args.config).read_text())
    cfg["debug"]["dir"] = str(ROOT / cfg["debug"]["dir"])
    load_env(ROOT / ".env")
    setup_logging(cfg["log"])
    set_mic_boost(cfg["audio"]["mic_boost_db"], cfg["audio"]["card"])
    asyncio.run(check(cfg) if args.check else run(cfg))


if __name__ == "__main__":
    main()
