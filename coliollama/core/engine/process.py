"""Subprocess wrapper around the native Colibrí engine (`coli serve`)."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


class EngineStartError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelTarget:
    name: str
    path: str


def find_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class EngineProcess:
    """One `coli serve` process bound to a single model for its whole life."""

    def __init__(
        self,
        command: tuple[str, ...],
        target: ModelTarget,
        port: int,
        log_path: Path,
        extra_args: tuple[str, ...] = (),
    ) -> None:
        self.target = target
        self.port = port
        self.started_at = time.time()
        self._command = [
            *command,
            "serve",
            "--model", target.path,
            "--model-id", target.name,
            "--host", "127.0.0.1",
            "--port", str(port),
            *extra_args,
        ]
        self._log_path = log_path
        self._proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    @property
    def log_path(self) -> Path:
        return self._log_path

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self._log_path, "ab")
        try:
            self._proc = subprocess.Popen(
                self._command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            raise EngineStartError(f"cannot launch engine {self._command[0]!r}: {exc}") from exc
        finally:
            log.close()
        self.started_at = time.time()

    def log_tail(self, lines: int = 15) -> str:
        try:
            text = self._log_path.read_text(errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def terminate(self, timeout: float = 20.0) -> None:
        """SIGTERM the whole process group, escalating to SIGKILL after `timeout`."""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        self._signal(proc, signal.SIGTERM)
        try:
            proc.wait(timeout)
        except subprocess.TimeoutExpired:
            self._signal(proc, getattr(signal, "SIGKILL", signal.SIGTERM))
            proc.wait()

    @staticmethod
    def _signal(proc: subprocess.Popen, sig: int) -> None:
        try:
            if hasattr(os, "killpg"):
                os.killpg(proc.pid, sig)
            else:
                proc.send_signal(sig)
        except (ProcessLookupError, PermissionError):
            pass
