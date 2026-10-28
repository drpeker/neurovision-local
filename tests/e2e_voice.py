"""Voice test in the running robot: questions are spoken through the speaker and heard by the mic.

    .venv/bin/python -m tests.e2e_voice config.dev.toml
"""
import os, re, signal, subprocess, sys, time, wave
from pathlib import Path

from piper import PiperVoice

ROOT = Path(__file__).resolve().parent.parent
LOG = Path("/tmp/nvl_e2e.log")
QUESTIONS = sys.argv[2].split("|") if len(sys.argv) > 2 else [
    "What is your name?", "What do you see?", "Turn left.", "Set the volume to 80.", "What time is it?",
    "Remember that my keys are in the blue bowl.", "Where are my keys?", "How hot are you?",
    "What is the capital of Finland?", "Tell me a joke.", "Who was Albert Einstein?",
    "What's the weather in Ankara?", "Follow me.", "Stop."]

voice = PiperVoice.load("/home/drpeker/neurobrain/tts/en_US-lessac-medium.onnx")
for i, q in enumerate(QUESTIONS):
    with wave.open(f"/tmp/nvl_q{i}.wav", "wb") as w:
        voice.synthesize_wav(q, w)

config = sys.argv[1] if len(sys.argv) > 1 else "config.dev.toml"
app = subprocess.Popen([str(ROOT / ".venv/bin/python"), "robot.py", "--config", config], cwd=ROOT,
                       stdout=LOG.open("w"), stderr=subprocess.STDOUT, start_new_session=True)
text = lambda: LOG.read_text(errors="replace")


def wait(pred, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred(text()):
            return True
        time.sleep(0.2)
    return False


t0 = time.monotonic()
print(f"startup: {'ready' if wait(lambda t: 'listening' in t, 180) else 'TIMEOUT'} after {time.monotonic() - t0:.1f} s")
for i, q in enumerate(QUESTIONS):
    n = text().count("turn done")
    subprocess.run(["aplay", "-q", f"/tmp/nvl_q{i}.wav"])
    if not wait(lambda t: t.count("turn done") > n, 90):
        print(f"  no answer: {q!r}")
    time.sleep(1.0)
os.killpg(app.pid, signal.SIGINT)
app.wait(timeout=30)

log = text()
heard = re.findall(r"heard ['\"](.*?)['\"] \(.*?confidence (-?[\d.]+)\) -> (\w+)", log)
answers = re.findall(r"answer \(([\d.]+) s after speech\): (.*)", log)
for (h, conf, tool), (sec, a) in zip(heard, answers):
    print(f"  {float(sec):5.1f}s  {tool:12} conf {conf:>5}  {h!r} -> {a!r}")
print(f"heard {len(heard)}/{len(QUESTIONS)}")
