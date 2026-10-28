"""Parakeet TDT 0.6B v3 (NVIDIA, 25 languages) as a small HTTP server that answers like whisper-server.

whisper.cpp builds libparakeet.so but no server for it, so this one loads the library with
ctypes and keeps the model in memory. POST /inference with a 16 kHz mono WAV in the "file"
field returns {"text", "segments": [{"avg_logprob"}]}, the fields nv.local_ai.SpeechToText reads.

    python -m nv.parakeet_server --lib .../libparakeet.so --model .../ggml-parakeet-tdt-0.6b-v3-q4_k.bin --port 8181
"""

import argparse
import ctypes as C
import io
import json
import math
import re
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np


class ContextParams(C.Structure):
    _fields_ = [("use_gpu", C.c_bool), ("gpu_device", C.c_int)]


LOG_CALLBACK = C.CFUNCTYPE(None, C.c_int, C.c_char_p, C.c_void_p)


class FullParams(C.Structure):
    _fields_ = [("strategy", C.c_int), ("n_threads", C.c_int), ("offset_ms", C.c_int),
                ("duration_ms", C.c_int), ("no_context", C.c_bool), ("audio_ctx", C.c_int)] + \
               [(f"cb{i}", C.c_void_p) for i in range(10)]


class Parakeet:
    def __init__(self, lib: str, model: str, threads: int):
        self._lib = L = C.CDLL(lib)
        L.parakeet_init_from_file_with_params.restype = C.c_void_p
        L.parakeet_init_from_file_with_params.argtypes = [C.c_char_p, ContextParams]
        L.parakeet_full_default_params_by_ref.restype = C.POINTER(FullParams)
        L.parakeet_full.argtypes = [C.c_void_p, FullParams, C.POINTER(C.c_float), C.c_int]
        L.parakeet_full_n_segments.argtypes = [C.c_void_p]
        L.parakeet_full_get_segment_text.restype = C.c_char_p
        L.parakeet_full_get_segment_text.argtypes = [C.c_void_p, C.c_int]
        L.parakeet_full_n_tokens.argtypes = [C.c_void_p, C.c_int]
        L.parakeet_full_get_token_p.restype = C.c_float
        L.parakeet_full_get_token_p.argtypes = [C.c_void_p, C.c_int, C.c_int]
        self._quiet = LOG_CALLBACK(lambda *a: None)  # kept: the library holds the pointer
        L.parakeet_log_set(self._quiet, None)
        self._ctx = L.parakeet_init_from_file_with_params(model.encode(), ContextParams(False, 0))
        if not self._ctx:
            raise RuntimeError(f"could not load {model}")
        self._params = L.parakeet_full_default_params_by_ref(0).contents
        self._params.n_threads = threads
        self._lock = threading.Lock()

    def transcribe(self, samples: np.ndarray) -> dict:
        L, ctx = self._lib, self._ctx
        audio = np.ascontiguousarray(samples, np.float32)
        with self._lock:
            if L.parakeet_full(ctx, self._params, audio.ctypes.data_as(C.POINTER(C.c_float)), len(audio)):
                raise RuntimeError("parakeet_full failed")
            segments, texts = [], []
            for s in range(L.parakeet_full_n_segments(ctx)):
                texts.append(L.parakeet_full_get_segment_text(ctx, s).decode("utf-8", "replace"))
                logs = [math.log(max(L.parakeet_full_get_token_p(ctx, s, t), 1e-6))
                        for t in range(L.parakeet_full_n_tokens(ctx, s))]
                segments.append({"avg_logprob": sum(logs) / len(logs) if logs else 0.0})
        return {"text": re.sub(r"\s+", " ", " ".join(texts)).strip(), "segments": segments}


def read_wav(body: bytes, content_type: str) -> np.ndarray:
    boundary = content_type.split("boundary=")[-1].encode()
    for part in body.split(b"--" + boundary):
        head, _, data = part.partition(b"\r\n\r\n")
        if b'name="file"' in head:
            with wave.open(io.BytesIO(data.rsplit(b"\r\n", 1)[0])) as w:
                pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
            return pcm.astype(np.float32) / 32768
    raise ValueError("no file field")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lib", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    a = p.parse_args()
    model = Parakeet(a.lib, a.model, a.threads)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code: int, obj: dict) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200, {"status": "ok"})

        def do_POST(self):
            try:
                body = self.rfile.read(int(self.headers["Content-Length"]))
                samples = read_wav(body, self.headers["Content-Type"])
                self._send(200, model.transcribe(samples) if len(samples) > 1600 else {"text": "", "segments": []})
            except Exception as e:
                self._send(500, {"error": str(e)})

    print(f"parakeet server ready on {a.host}:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
