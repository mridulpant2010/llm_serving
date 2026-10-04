"""A fake OpenAI/vLLM-compatible server for testing the harness without a GPU.

!!! Everything it returns is SIMULATED.  Its numbers come from a toy latency
model and must never be used as experimental results.  Rows produced against it
are tagged ``backend: "mock"`` and written to a separate file by the CLI.

The toy model, for batch size B and ``k`` speculative tokens:

    step_time(B, w) = T_MEM + T_COMPUTE * B * w          (memory-bound at small B,
                                                           compute-bound at large B)
    base:   w = 1       -> 1 token per step
    spec:   w = 1 + k   -> E[tokens] = sum_{i=0..k} a^i per step

so speculation wins at small B and loses past a crossover, which gives the
analysis code something with a known shape to find.
"""

from __future__ import annotations

import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .config import ServerRun

T_MEM = 0.020  # s per step, independent of batch (weight/KV loading)
T_COMPUTE = 0.002  # s per token per sequence in the batch
BLOCK_SIZE = 16
ACCEPT = {"none": 0.0, "ngram": 0.35, "draft_model": 0.7, "eagle": 0.8}


class _State:
    def __init__(self, run: ServerRun, time_scale: float, base_blocks: int):
        self.run = run
        self.time_scale = time_scale
        self.lock = threading.Lock()
        self.inflight = 0
        self.kv_tokens = 0
        self.preemptions = 0
        self.draft_tokens = 0
        self.accepted_tokens = 0
        self.num_drafts = 0
        spec = run.spec_config or {}
        self.k = int(spec.get("num_speculative_tokens", 0)) if run.spec_name != "none" else 0
        self.accept = ACCEPT.get(run.spec_name, 0.5)
        blocks = base_blocks * (2 if run.kv_dtype.startswith("fp8") else 1)
        if run.spec_name == "draft_model":
            blocks = int(blocks * 0.9)  # drafter weights take VRAM from the KV pool
        self.num_blocks = blocks

    def kv_usage(self) -> float:
        return min(1.0, (self.kv_tokens / BLOCK_SIZE) / self.num_blocks)


def _make_handler(state: _State):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence default stderr logging
            pass

        def _json(self, code: int, obj) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self._json(200, {"data": [{"id": "mock"}]})
            elif self.path == "/metrics":
                s = state
                with s.lock:
                    text = (
                        f'vllm:num_requests_running{{model_name="mock"}} {s.inflight}\n'
                        f'vllm:num_requests_waiting{{model_name="mock"}} 0\n'
                        f'vllm:gpu_cache_usage_perc{{model_name="mock"}} {s.kv_usage()}\n'
                        f'vllm:num_preemptions_total{{model_name="mock"}} {s.preemptions}\n'
                        f'vllm:spec_decode_num_draft_tokens_total{{model_name="mock"}} {s.draft_tokens}\n'
                        f'vllm:spec_decode_num_accepted_tokens_total{{model_name="mock"}} {s.accepted_tokens}\n'
                        f'vllm:spec_decode_num_drafts_total{{model_name="mock"}} {s.num_drafts}\n'
                        f'vllm:cache_config_info{{block_size="{BLOCK_SIZE}",'
                        f'cache_dtype="{s.run.kv_dtype}",num_gpu_blocks="{s.num_blocks}"}} 1.0\n'
                    )
                body = text.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/completions":
                return self._json(404, {"error": "not found"})
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if payload.get("prompt_logprobs") is not None:
                return self._perplexity_reply(payload)
            self._stream_reply(payload)

        def _perplexity_reply(self, payload):
            ids = payload["prompt"]
            penalty = 0.01 if state.run.kv_dtype.startswith("fp8") else 0.0
            plp = [None] + [
                {str(t): {"logprob": -2.0 - penalty + 0.1 * ((i % 7) - 3) / 3, "rank": 1}}
                for i, t in enumerate(ids[1:], start=1)
            ]
            self._json(200, {"choices": [{"text": "x", "prompt_logprobs": plp}]})

        def _stream_reply(self, payload):
            s = state
            n_prompt, max_tokens = len(payload["prompt"]), int(payload["max_tokens"])
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()  # HTTP/1.0: connection closes at end of stream
            rng = random.Random()
            with s.lock:
                s.inflight += 1
                s.kv_tokens += n_prompt
            produced = 0
            try:
                while produced < max_tokens:
                    with s.lock:
                        batch = s.inflight
                        over = (s.kv_tokens / BLOCK_SIZE) > s.num_blocks
                        if over:
                            s.preemptions += 1
                    width = 1 + s.k
                    step = T_MEM + T_COMPUTE * batch * width
                    if over:
                        step *= 2.0  # crude cost of preempt + recompute
                    time.sleep(step * s.time_scale)
                    emitted = 1
                    if s.k:
                        acc = 0
                        while acc < s.k and rng.random() < s.accept:
                            acc += 1
                        emitted += acc
                        with s.lock:
                            s.draft_tokens += s.k
                            s.accepted_tokens += acc
                            s.num_drafts += 1
                    emitted = min(emitted, max_tokens - produced)
                    produced += emitted
                    with s.lock:
                        s.kv_tokens += emitted
                    chunk = {"choices": [{"text": " tok" * emitted}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                usage = {"choices": [], "usage": {"prompt_tokens": n_prompt, "completion_tokens": produced}}
                self.wfile.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with s.lock:
                    s.inflight -= 1
                    s.kv_tokens -= n_prompt + produced

    return Handler


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 1024


class MockServer:
    """Same interface as ``VllmServer``: a context manager with ``base_url``."""

    def __init__(self, run: ServerRun, time_scale: float = 0.25, base_blocks: int = 2000):
        self._state = _State(run, time_scale, base_blocks)
        self._httpd = _Server(("127.0.0.1", 0), _make_handler(self._state))
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def __enter__(self) -> "MockServer":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
