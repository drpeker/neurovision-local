"""NeuroVision local: voice and vision robot with an on-device brain (Qwen 1.7B, Whisper, Piper).

    .venv/bin/python robot.py                          # run the robot (robot hardware)
    .venv/bin/python robot.py --config config.dev.toml # Pi 4 bench: USB camera and mic
    .venv/bin/python robot.py --check                  # hardware test: mic, speaker, camera
"""

import argparse
import asyncio
import logging
import logging.handlers
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from nv.audio import Mic, Speaker, set_mic_boost
from nv.camera import Camera
from nv.vad import SpeechGate

ROOT = Path(__file__).resolve().parent
log = logging.getLogger("robot")


def setup_logging(cfg: dict) -> None:
    path = ROOT / cfg["file"]
    path.parent.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    file_h = logging.handlers.RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3)
    console_h = logging.StreamHandler(sys.stdout)
    for h in (file_h, console_h):
        h.setFormatter(fmt)
    logging.basicConfig(level=cfg["level"], handlers=[file_h, console_h])


def make_camera(cfg: dict) -> Camera | None:
    c = cfg["camera"]
    if not c["enabled"]:
        return None
    return Camera(c["width"], c["height"], c["fps"], c["rotation"], c.get("hflip", False),
                  c.get("backend", "rpicam"), c.get("device", "/dev/video0"))


async def check(cfg: dict) -> None:
    """Record 3 s, count speech frames, play it back, grab a camera frame."""
    a, v = cfg["audio"], cfg["vad"]
    mic, speaker = Mic(a["mic"], a["highpass_hz"]), Speaker(a["speaker"], a["echo_tail_ms"])
    gate = SpeechGate(v["aggressiveness"], v["preroll_ms"], v["hangover_ms"])
    camera = make_camera(cfg)
    if camera:
        await camera.start()
    mic.start()
    print("Speak for 3 seconds...")
    frames, voiced = [], 0
    while len(frames) < 150:
        f = await mic.frames.get()
        frames.append(f)
        voiced += gate._vad.is_speech(f, 16000)
    mic.stop()
    peak = max(abs(int.from_bytes(f[i:i + 2], "little", signed=True))
               for f in frames for i in range(0, len(f), 2))
    print(f"microphone: {len(frames)} frames, speech in {voiced}, peak {peak / 32768:.2f}")

    # 16 kHz -> 24 kHz: every 2 samples become 3 (rough, test only)
    pcm16 = b"".join(frames)
    out = bytearray()
    for i in range(0, len(pcm16) - 3, 4):
        s0, s1 = pcm16[i:i + 2], pcm16[i + 2:i + 4]
        out += s0 + s0 + s1
    speaker.start()
    print("playing the recording back...")
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
            print(f"camera: {len(jpeg)} byte JPEG -> logs/check.jpg")
        else:
            print("camera: no frame")
        await camera.stop()


async def run(cfg: dict) -> None:
    from nv.brain import LocalBrain
    from nv.local_ai import Servers
    from nv.memory import Memory

    started = time.monotonic()
    # The servers load their models while the robot opens its hardware
    servers = Servers(cfg["local"], ROOT / "logs")
    servers.start()
    memory = Memory(ROOT / cfg["memory"]["db"])
    log.info("memory: %s", memory.count())
    a = cfg["audio"]
    volume = int(memory.get("volume") or a["volume"])  # a spoken volume change is kept
    if a.get("card") and memory.get("mic_boost_db"):
        set_mic_boost(int(memory.get("mic_boost_db")), a["card"])  # so is a spoken microphone change
    mic = Mic(a["mic"], a["highpass_hz"])
    speaker = Speaker(a["speaker"], a["echo_tail_ms"], volume)
    camera = make_camera(cfg)
    task = None
    try:
        if camera:
            await camera.start()
        speaker.start()
        mic.start()
        brain = LocalBrain(cfg, mic, speaker, camera, memory)
        face = await make_face(cfg, speaker, brain)
        motors = make_motors(cfg, brain, memory)
        imu = await make_imu(cfg, motors, brain)
        tracker = make_vision(cfg, camera, motors, brain, face)
        # Commands need only Whisper (~12 s); Qwen takes ~36 s to load from the SD card and is
        # only needed for chat, so the robot starts listening without waiting for it
        await servers.wait_ready(("STT",))
        log.info("speech recognition loaded in %.1f s", time.monotonic() - started)

        task = asyncio.create_task(brain.run(servers.wait_ready(("LLM",))), name="brain")
        stop_target = []

        def on_shutdown(target: str) -> None:
            stop_target.append(target)
            task.cancel()

        brain.attach_shutdown(on_shutdown)
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, task.cancel)
        log.info("NeuroVision local started (%.1f s), waiting for the language model. Ctrl+C to stop.",
                 time.monotonic() - started)
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            if tracker:
                await tracker.stop()
            if motors:
                motors.close()  # motors first: always stop
            if imu:
                imu.stop()
            if face:
                await face.stop()
    finally:
        mic.stop()
        speaker.stop()
        if camera:
            await camera.stop()
        servers.stop()
        memory.close()
        log.info("stopped")
    if task and stop_target == ["sistem"]:
        # needs the sudoers rule from deploy/allow-poweroff.sh
        r = subprocess.run(["sudo", "-n", "systemctl", "poweroff"], capture_output=True, text=True)
        if r.returncode:
            log.warning("could not power off the Pi (was deploy/allow-poweroff.sh run?): %s", r.stderr.strip())


def make_vision(cfg: dict, camera, motors, brain, face):
    """Object recognition for 'look'; tracking as well when the motors are there."""
    v = cfg["vision"]
    model = ROOT / v["model"]
    if not (v["enabled"] and camera):
        return None
    if not model.exists():
        log.warning("object recognition off: model missing (%s); deploy/get-model.sh", model)
        return None
    from nv.vision import Detector
    detector = Detector(model, v["threads"], v["threshold"])
    brain.attach_detector(detector)
    if not motors:
        return None
    from nv.tracker import Tracker
    tracker = Tracker(v, camera, motors, detector)
    tracker.face = face
    brain.attach_tracker(tracker)
    return tracker


def make_motors(cfg: dict, brain, memory):
    """If the motor driver does not open, the robot keeps running without moving."""
    if not cfg["motors"]["enabled"]:
        return None
    from nv.motors import Motors
    try:
        motors = Motors(cfg["motors"])
    except Exception as e:
        log.warning("motors did not open, continuing without them: %s", e)
        return None
    saved = memory.get("motor_speed")  # a spoken speed change is kept
    if saved:
        motors.speed_level = int(saved)
    brain.attach_motors(motors)
    return motors


async def make_imu(cfg: dict, motors, brain):
    """MPU6050: turns by angle and straight driving that holds its heading. Without it (or if it
    does not answer) the robot moves by time and power as before."""
    if not (cfg.get("imu", {}).get("enabled") and motors):
        return None
    from nv.imu import AngleTurner, Imu
    try:
        imu = Imu(cfg["imu"]["bus"])
        await asyncio.to_thread(imu.calibrate)  # the robot is standing still at start
    except Exception as e:
        log.warning("MPU6050 did not answer, moving by time and power: %s", e)
        return None
    imu.start()
    motors.imu = imu
    brain.attach_turner(AngleTurner(imu, motors, cfg["imu"]))
    return imu


async def make_face(cfg: dict, speaker: Speaker, brain):
    """Without a screen (or if it fails) the robot keeps running without a face."""
    if not cfg["display"]["enabled"]:
        return None
    from nv.display import Face, Panel
    try:
        panel = Panel(cfg["display"])
    except Exception as e:
        log.warning("screen did not open, continuing without it: %s", e)
        return None
    face = Face(cfg["display"], panel, speaker, brain.gate, brain.body, wait_ready=True)
    await face.start()
    brain.attach_face(face)
    asyncio.create_task(show_wifi_setup(face), name="wifi-status")
    return face


async def show_wifi_setup(face) -> None:
    """While the Wi-Fi setup service (deploy/wifi) runs, keep its instructions on screen."""
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
    p.add_argument("--check", action="store_true", help="hardware test")
    p.add_argument("--config", default=str(ROOT / "config.toml"))
    args = p.parse_args()

    cfg = tomllib.loads(Path(args.config).read_text())
    cfg["debug"]["dir"] = str(ROOT / cfg["debug"]["dir"])
    setup_logging(cfg["log"])
    if cfg["audio"].get("card"):
        set_mic_boost(cfg["audio"]["mic_boost_db"], cfg["audio"]["card"])
    asyncio.run(check(cfg) if args.check else run(cfg))


if __name__ == "__main__":
    main()
