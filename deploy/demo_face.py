"""Ekran onizlemesi: robotun yuzundeki tum durumlari ve duygulari sirayla gosterir.

    .venv/bin/python deploy/demo_face.py      (robot servisi durdurulmus olmali)
"""
import asyncio
import os
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nv.display import EMOTIONS, Face, Panel  # noqa: E402


class Fake:
    busy = False
    active = False
    connected_since = 1.0


async def main():
    cfg = tomllib.loads((ROOT / "config.toml").read_text())["display"]
    speaker, gate, body = Fake(), Fake(), Fake()
    face = Face(cfg, Panel(cfg), speaker, gate, body)
    await face.start()
    cpu0, t0 = os.times(), time.monotonic()
    steps = [("bosta (goz kirpma, etrafa bakma)", {}), ("dinliyor", {"gate": True}),
             ("dusunuyor", {"thinking": True}), ("konusuyor", {"speaker": True}),
             ("bakiyor (look)", {"flash": True}), ("baglanti yok", {"offline": True})]
    steps += [(f"duygu: {e}", {"emotion": e}) for e in EMOTIONS if e != "nötr"]
    for name, s in steps:
        print(name, flush=True)
        gate.active, speaker.busy = s.get("gate", False), s.get("speaker", False)
        face.thinking = s.get("thinking", False)
        body.connected_since = None if s.get("offline") else 1.0
        if s.get("emotion"):
            face.set_emotion(s["emotion"], 3)
        end = time.monotonic() + 3
        while time.monotonic() < end:
            if s.get("flash"):
                face.flash()
            await asyncio.sleep(0.3 if s.get("flash") else 0.1)
    cpu1 = os.times()
    wall = time.monotonic() - t0
    print(f"ortalama CPU: %{100 * (cpu1.user + cpu1.system - cpu0.user - cpu0.system) / wall:.0f} (tek cekirdek)")
    await face.stop()


asyncio.run(main())
