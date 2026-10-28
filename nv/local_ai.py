"""Local AI services: Qwen (llama-server), Parakeet (nv/parakeet_server.py) and Piper, all on the Pi.

The two servers run as child processes and are stopped with the robot. Settings live in
config.toml [local]; numbers in comments were measured on a Pi 4 (4 GB, Cortex-A72).
"""

import asyncio
import io
import json
import logging
import re
import subprocess
import sys
import time
import unicodedata
import urllib.request
import uuid
import wave
from pathlib import Path

import numpy as np

from .audio import MIC_RATE, SPEAKER_RATE

log = logging.getLogger(__name__)

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class Servers:
    """Starts llama-server and the Parakeet server; both load while the robot opens its hardware."""

    def __init__(self, cfg: dict, log_dir: Path):
        self._cfg = cfg
        self._log_dir = log_dir
        self._procs: list[subprocess.Popen] = []

    def start(self) -> None:
        c = self._cfg
        self._log_dir.mkdir(exist_ok=True)
        # Leftovers from a crashed run would hold the ports (only ours: NeuroBrain uses other ports)
        for port in (c["llm_port"], c["stt_port"]):
            subprocess.run(["pkill", "-f", f"--port {port}$"], capture_output=True)
        time.sleep(0.5)
        self._spawn("llm", [
            c["llama_server"], "-m", c["llm_model"], "--reasoning", "off",
            "-c", str(c["llm_context"]), "-t", str(c["llm_threads"]),
            # One slot: the system prompt is read once and reused from the cache
            "-np", "1", "--host", "127.0.0.1", "--port", str(c["llm_port"]),
        ])
        self._spawn("stt", [
            # Parakeet answers like whisper-server (same /inference, same fields); it needs no VAD
            # pass of its own: SpeechGate already cut the speech out
            sys.executable, "-m", "nv.parakeet_server", "--lib", c["parakeet_lib"], "--model", c["stt_model"],
            "--threads", str(c["stt_threads"]), "--port", str(c["stt_port"]),
        ], need=(c["parakeet_lib"], c["stt_model"]))

    def _spawn(self, name: str, args: list[str], need: tuple[str, ...] = ()) -> None:
        for path in need or (args[0], args[2]):
            if not Path(path).exists():
                raise FileNotFoundError(f"{name}: {path} is missing (see config.toml [local])")
        out = open(self._log_dir / f"{name}-server.log", "w")
        self._procs.append(subprocess.Popen(args, stdout=out, stderr=subprocess.STDOUT,
                                            cwd=Path(__file__).resolve().parent.parent))
        log.info("%s server starting: %s", name, Path(need[-1] if need else args[2]).name)

    async def wait_ready(self, which: tuple[str, ...] = ("STT", "LLM"), timeout: float = 120) -> None:
        c = self._cfg
        urls = {"STT": f"http://127.0.0.1:{c['stt_port']}/", "LLM": f"http://127.0.0.1:{c['llm_port']}/health"}
        for name, url in ((n, urls[n]) for n in which):
            deadline = time.monotonic() + timeout
            while True:
                if any(p.poll() is not None for p in self._procs):
                    raise RuntimeError(f"a local AI server exited; see {self._log_dir}/*-server.log")
                try:
                    await asyncio.to_thread(urllib.request.urlopen, url, None, 1)
                    break
                except Exception:
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"{name} server did not start within {timeout:.0f} s")
                    await asyncio.sleep(0.3)
            log.info("%s server ready", name)

    def stop(self) -> None:
        for p in self._procs:
            if p.poll() is None:
                p.terminate()
        for p in self._procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        self._procs.clear()


class SpeechToText:
    """Parakeet over HTTP (whisper-server's interface), audio sent from memory. Returns (text, confidence).

    confidence is the lowest segment mean token log-probability: ~0 is certain. On the robot
    microphone correct English sentences scored -0.00 to -0.22, a misheard "Stop." -0.68.
    Parakeet v3 picks the language itself; a sentence it writes in another script (Cyrillic)
    was misheard, so it gets the lowest confidence and the robot asks again.
    """

    def __init__(self, cfg: dict):
        self._url = f"http://127.0.0.1:{cfg['stt_port']}/inference"
        self._language = cfg["language"]

    async def transcribe(self, pcm: bytes) -> tuple[str, float]:
        return await asyncio.to_thread(self._transcribe, pcm)

    def _transcribe(self, pcm: bytes) -> tuple[str, float]:
        wav = io.BytesIO()
        with wave.open(wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(MIC_RATE)
            w.writeframes(pcm)
        boundary = uuid.uuid4().hex
        body = io.BytesIO()
        for name, value in {"response_format": "verbose_json", "language": self._language}.items():
            body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
        body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="speech.wav"\r\n'
                   f"Content-Type: audio/wav\r\n\r\n".encode())
        body.write(wav.getvalue())
        body.write(f"\r\n--{boundary}--\r\n".encode())
        req = urllib.request.Request(self._url, data=body.getvalue(),
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            result = json.load(r)
        scores = [s.get("avg_logprob", 0.0) for s in result.get("segments", [])]
        text = dedupe(result.get("text", "").strip())
        if any(c.isalpha() and not c.isascii() and "LATIN" not in unicodedata.name(c, "") for c in text):
            return text, -9.0
        return text, min(scores, default=0.0)


def dedupe(text: str) -> str:
    """A recogniser sometimes repeats a sentence ("Is it on? Is it on? Is it on?"): keep one."""
    out: list[str] = []
    for sentence in SENTENCE_END.split(text):
        if not out or sentence.lower().strip(" .!?") != out[-1].lower().strip(" .!?"):
            out.append(sentence)
    return " ".join(out)


class LanguageModel:
    """Qwen chat, streamed: each finished sentence is handed over while the rest is generated."""

    def __init__(self, cfg: dict, system: str):
        self._url = f"http://127.0.0.1:{cfg['llm_port']}/v1/chat/completions"
        self._system = system
        self._max_tokens = cfg["llm_max_tokens"]
        # The last exchange (question as sent, answer, time) goes with the next question for this long
        self._keep_s = cfg.get("llm_context_s", 0)
        self._last: tuple[str, str, float] | None = None
        self.timings: dict = {}

    def _post(self, messages: list[dict], max_tokens: int, stream: bool):
        data = {"messages": messages, "max_tokens": max_tokens, "stream": stream, "cache_prompt": True}
        req = urllib.request.Request(self._url, data=json.dumps(data).encode(),
                                     headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=180)

    async def warm_up(self) -> float:
        """Read the system prompt into the cache now, so the first question is not slower."""
        t = time.monotonic()
        msgs = [{"role": "system", "content": self._system}, {"role": "user", "content": "Hi"}]
        await asyncio.to_thread(lambda: self._post(msgs, 1, False).read())
        return time.monotonic() - t

    async def chat(self, text: str, on_sentence) -> tuple[str, bool]:
        """Returns (spoken answer, cut at the token limit). on_sentence(str) is called per sentence."""
        loop = asyncio.get_running_loop()
        msgs = [{"role": "system", "content": self._system}]
        if self._last and time.monotonic() - self._last[2] < self._keep_s:
            # Sent exactly as last time, so llama-server reuses its cache for it
            msgs += [{"role": "user", "content": self._last[0]}, {"role": "assistant", "content": self._last[1]}]
        msgs.append({"role": "user", "content": text})

        def run() -> tuple[str, bool]:
            pending, spoken, finish = "", [], None
            with self._post(msgs, self._max_tokens, True) as r:
                for raw in r:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    self.timings = chunk.get("timings") or self.timings
                    if not chunk.get("choices"):
                        continue
                    choice = chunk["choices"][0]
                    finish = choice.get("finish_reason") or finish
                    pending += choice.get("delta", {}).get("content") or ""
                    parts = SENTENCE_END.split(pending)
                    for sentence in parts[:-1]:
                        spoken.append(sentence.strip())
                        loop.call_soon_threadsafe(on_sentence, sentence.strip())
                    pending = parts[-1]
            # At the token limit mid-sentence: drop the fragment unless it is all there is
            cut = finish == "length" and bool(spoken)
            if pending.strip() and not cut:
                spoken.append(pending.strip())
                loop.call_soon_threadsafe(on_sentence, pending.strip())
            return " ".join(spoken), cut

        self.timings = {}
        answer, cut = await asyncio.to_thread(run)
        self._last = (text, answer, time.monotonic()) if answer else None
        return answer, cut


class TextToSpeech:
    """Piper, resampled to the speaker's 24 kHz (Piper voices are 22.05 kHz)."""

    def __init__(self, model: str):
        from piper import PiperVoice
        self._voice = PiperVoice.load(model)
        self._rate = self._voice.config.sample_rate
        log.info("voice loaded: %s (%d Hz)", Path(model).name, self._rate)

    async def synth(self, text: str) -> bytes:
        return await asyncio.to_thread(self._synth, text)

    def _synth(self, text: str) -> bytes:
        audio = np.concatenate([np.frombuffer(c.audio_int16_bytes, np.int16)
                                for c in self._voice.synthesize(text)] or [np.zeros(0, np.int16)])
        if self._rate == SPEAKER_RATE or not len(audio):
            return audio.tobytes()
        n = int(len(audio) * SPEAKER_RATE / self._rate)
        out = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio.astype(np.float32))
        return out.astype(np.int16).tobytes()
