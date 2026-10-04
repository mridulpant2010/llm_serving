"""Launch and manage a vLLM OpenAI-compatible server for one experiment run.

Flag names follow recent vLLM releases (``--speculative-config`` as JSON,
``--no-enable-prefix-caching``).  vLLM changes its CLI often, so pin one version
for the whole project and adjust ``build_vllm_command`` if your version differs.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from .config import ExperimentConfig, ServerRun


def build_vllm_command(
    cfg: ExperimentConfig, run: ServerRun, port: int, host: str = "127.0.0.1"
) -> list[str]:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", cfg.model,
        "--host", host,
        "--port", str(port),
        "--dtype", cfg.dtype,
        "--kv-cache-dtype", run.kv_dtype,
        "--max-model-len", str(run.max_model_len),
        # Without this the scheduler silently caps in-flight requests at its
        # default and the concurrency sweep would measure the cap, not memory.
        "--max-num-seqs", str(run.max_num_seqs),
        "--gpu-memory-utilization", str(cfg.fixed.gpu_memory_utilization),
        "--seed", str(cfg.fixed.seed),
        "--enable-prefix-caching" if run.prefix_caching else "--no-enable-prefix-caching",
    ]
    if cfg.tokenizer:
        cmd += ["--tokenizer", cfg.tokenizer]
    if cfg.quantization:
        cmd += ["--quantization", cfg.quantization]
    if cfg.fixed.enforce_eager:
        cmd.append("--enforce-eager")
    if run.spec_config:
        cmd += ["--speculative-config", json.dumps(run.spec_config)]
    cmd += list(cfg.fixed.extra_server_args)
    return cmd


class ServerStartError(RuntimeError):
    pass


class VllmServer:
    """Context manager: start, wait until healthy, always shut down."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        run: ServerRun,
        port: int = 8000,
        log_dir: str | Path = "results/server_logs",
    ):
        self.cfg, self.run, self.port = cfg, run, port
        self.log_path = Path(log_dir) / f"{run.run_id}.log"
        self._proc: subprocess.Popen | None = None
        self._log = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "w", encoding="utf-8")
        cmd = build_vllm_command(self.cfg, self.run, self.port)
        self._log.write("$ " + " ".join(cmd) + "\n")
        self._log.flush()
        kwargs = {"start_new_session": True} if os.name == "posix" else {}
        self._proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT, **kwargs)
        self._wait_ready(self.cfg.fixed.server_start_timeout_s)

    def _wait_ready(self, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        with httpx.Client(timeout=5.0, trust_env=False) as client:
            while time.time() < deadline:
                if self._proc.poll() is not None:
                    raise ServerStartError(
                        f"vLLM exited with code {self._proc.returncode}.\n{self._tail()}"
                    )
                try:
                    if client.get(f"{self.base_url}/health").status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(2.0)
        raise ServerStartError(f"vLLM not healthy after {timeout_s}s.\n{self._tail()}")

    def _tail(self, n: int = 25) -> str:
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            return f"--- last {n} lines of {self.log_path} ---\n" + "\n".join(lines[-n:])
        except OSError:
            return "(no log available)"

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                else:
                    self._proc.terminate()
                self._proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                else:
                    self._proc.kill()
                self._proc.wait()
        if self._log:
            self._log.close()
        time.sleep(3.0)  # let the driver release GPU memory before the next launch

    def __enter__(self) -> "VllmServer":
        try:
            self.start()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
