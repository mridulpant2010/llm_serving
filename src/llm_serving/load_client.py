"""Async load client: streams completions and timestamps every request.

Closed loop  : keeps exactly ``concurrency`` requests in flight at all times.
               This is what the concurrency sweep uses (C is the x-axis).
Open loop    : Poisson arrivals at a fixed rate, independent of completions.

Latency definitions (per request):
  TTFT = first text chunk - request submitted
  TPOT = (last text chunk - first text chunk) / (output_tokens - 1)
  E2E  = stream finished - request submitted

``output_tokens`` comes from the server's ``usage`` field, NOT from counting
chunks.  With speculative decoding one chunk can carry several tokens, so
counting chunks would make speculation look slower than it is.

Limitation: everything runs in one asyncio loop.  At very high concurrency the
client itself can become the bottleneck (JSON parsing per chunk); sanity-check
that client CPU stays below 100% when interpreting results at C >= 128.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass

import httpx
import numpy as np

from .workload import Request


@dataclass
class RequestResult:
    ok: bool
    error: str | None
    submit: float
    first: float | None
    last: float | None
    end: float
    n_prompt: int
    n_output: int

    @property
    def ttft(self) -> float | None:
        return None if self.first is None else self.first - self.submit

    @property
    def tpot(self) -> float | None:
        if self.first is None or self.last is None or self.n_output < 2:
            return None
        return (self.last - self.first) / (self.n_output - 1)

    @property
    def e2e(self) -> float:
        return self.end - self.submit


@dataclass
class LoadResult:
    results: list[RequestResult]
    wall_time: float


async def _one_request(
    client: httpx.AsyncClient, model: str, req: Request
) -> RequestResult:
    payload = {
        "model": model,
        "prompt": req.prompt_token_ids,
        "max_tokens": req.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,  # always generate exactly max_tokens (vLLM extension)
    }
    submit = time.perf_counter()
    first = last = None
    n_chunks = 0
    n_prompt = len(req.prompt_token_ids)
    n_output: int | None = None
    try:
        async with client.stream("POST", "/v1/completions", json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "replace")[:200]
                return RequestResult(
                    False, f"HTTP {resp.status_code}: {body}", submit, None, None,
                    time.perf_counter(), n_prompt, 0,
                )
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                usage = obj.get("usage")
                if usage:
                    n_output = usage.get("completion_tokens", n_output)
                    n_prompt = usage.get("prompt_tokens", n_prompt)
                choices = obj.get("choices") or []
                if choices and choices[0].get("text"):
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now
                    n_chunks += 1
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        return RequestResult(
            False, f"{type(e).__name__}: {e}", submit, first, last,
            time.perf_counter(), n_prompt, n_chunks,
        )
    end = time.perf_counter()
    if first is None:
        return RequestResult(False, "no tokens received", submit, None, None, end, n_prompt, 0)
    return RequestResult(
        True, None, submit, first, last, end, n_prompt,
        n_output if n_output is not None else n_chunks,
    )


def _client(base_url: str, timeout_s: float, max_conn: int) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(timeout_s, connect=30.0),
        limits=httpx.Limits(max_connections=max_conn, max_keepalive_connections=max_conn),
        trust_env=False,  # never route localhost traffic through a proxy
    )


async def run_closed_loop(
    base_url: str,
    model: str,
    requests: list[Request],
    concurrency: int,
    timeout_s: float = 900.0,
) -> LoadResult:
    queue: asyncio.Queue[Request] = asyncio.Queue()
    for r in requests:
        queue.put_nowait(r)
    results: list[RequestResult] = []

    async with _client(base_url, timeout_s, concurrency + 8) as client:

        async def worker() -> None:
            while True:
                try:
                    req = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                results.append(await _one_request(client, model, req))

        t0 = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(min(concurrency, len(requests)))])
        wall = time.perf_counter() - t0
    return LoadResult(results, wall)


async def run_open_loop(
    base_url: str,
    model: str,
    requests: list[Request],
    rate_per_s: float,
    seed: int = 0,
    timeout_s: float = 900.0,
) -> LoadResult:
    rng = random.Random(seed)
    results: list[RequestResult] = []

    async with _client(base_url, timeout_s, len(requests) + 8) as client:

        async def fire(req: Request) -> None:
            results.append(await _one_request(client, model, req))

        t0 = time.perf_counter()
        tasks = []
        for req in requests:
            tasks.append(asyncio.create_task(fire(req)))
            await asyncio.sleep(rng.expovariate(rate_per_s))
        await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0
    return LoadResult(results, wall)


def _pcts(values: list[float], prefix: str, qs=(50, 90, 99)) -> dict[str, float | None]:
    if not values:
        return {f"{prefix}_p{q}": None for q in qs}
    arr = np.asarray(values) * 1000.0  # seconds -> ms
    return {f"{prefix}_p{q}": float(np.percentile(arr, q)) for q in qs}


def summarize(load: LoadResult) -> dict:
    """Collapse per-request results into one metrics row."""
    ok = [r for r in load.results if r.ok]
    failed = len(load.results) - len(ok)
    out_tokens = sum(r.n_output for r in ok)
    row: dict = {
        "num_requests": len(load.results),
        "num_failed": failed,
        "duration_s": load.wall_time,
        "output_tokens_total": out_tokens,
        "throughput_tps": out_tokens / load.wall_time if load.wall_time > 0 else None,
        "req_throughput": len(ok) / load.wall_time if load.wall_time > 0 else None,
        "first_error": next((r.error for r in load.results if not r.ok), None),
    }
    row.update(_pcts([r.ttft for r in ok if r.ttft is not None], "ttft_ms"))
    row.update(_pcts([r.tpot for r in ok if r.tpot is not None], "tpot_ms"))
    row.update(_pcts([r.e2e for r in ok], "e2e_ms", qs=(50, 99)))
    return row
