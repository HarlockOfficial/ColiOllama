"""Engine lifecycle: startup, health-checks, graceful shutdown and cleanup."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import time
from pathlib import Path

import httpx

from coliollama.core.config import Settings
from coliollama.core.engine import gpu
from coliollama.core.engine.process import (
    EngineProcess,
    EngineStartError,
    ModelTarget,
    find_free_port,
)


class EngineLifecycle:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._engine: EngineProcess | None = None
        self._processor = "cpu"
        self._lock = asyncio.Lock()

    @property
    def engine(self) -> EngineProcess | None:
        return self._engine if self._engine and self._engine.is_alive() else None

    @property
    def active_model(self) -> str | None:
        engine = self.engine
        return engine.target.name if engine else None

    @property
    def base_url(self) -> str:
        engine = self.engine
        if engine is None:
            raise EngineStartError("engine is not running")
        return engine.base_url

    def info(self) -> dict | None:
        engine = self.engine
        if engine is None:
            return None
        return {
            "model": engine.target.name,
            "path": engine.target.path,
            "pid": engine.pid,
            "port": engine.port,
            "started_at": engine.started_at,
            "processor": self._processor,
        }

    async def start(self, target: ModelTarget) -> None:
        """Boot the engine on `target`, shutting down any previous engine first."""
        async with self._lock:
            await self._stop_locked()
            command = self._settings.resolve_coli()
            if not command:
                raise EngineStartError(
                    "Colibrí `coli` launcher not found: set COLIOLLAMA_COLI, COLIBRI_HOME or put `coli` on PATH"
                )
            if not Path(target.path).exists():
                raise EngineStartError(f"model path does not exist: {target.path}")
            gpu_args, self._processor = await asyncio.to_thread(
                gpu.engine_gpu_args, self._settings.gpu, command, target.path, self._settings.engine_args
            )
            engine = EngineProcess(
                command,
                target,
                find_free_port(),
                self._settings.log_dir / "engine.log",
                (*gpu_args, *self._settings.engine_args),
            )
            engine.start()
            self._engine = engine
            try:
                await self._wait_healthy(engine)
            except BaseException:
                await asyncio.to_thread(engine.terminate, self._settings.shutdown_timeout)
                self._engine = None
                raise
            self._write_state(engine)

    async def stop(self) -> bool:
        async with self._lock:
            return await self._stop_locked()

    async def _stop_locked(self) -> bool:
        engine, self._engine = self._engine, None
        if engine is None:
            return False
        was_alive = engine.is_alive()
        await asyncio.to_thread(engine.terminate, self._settings.shutdown_timeout)
        self._clear_state()
        return was_alive

    async def _wait_healthy(self, engine: EngineProcess) -> None:
        deadline = time.monotonic() + self._settings.startup_timeout
        delay = 0.2
        async with httpx.AsyncClient(timeout=5.0) as client:
            while True:
                if not engine.is_alive():
                    raise EngineStartError(
                        f"engine exited during startup for {engine.target.name!r}:\n{engine.log_tail()}"
                    )
                try:
                    resp = await client.get(f"{engine.base_url}/health")
                    if resp.status_code == 200 and resp.json().get("status") == "ok":
                        return
                except (httpx.HTTPError, ValueError):
                    pass
                if time.monotonic() > deadline:
                    raise EngineStartError(
                        f"engine for {engine.target.name!r} not healthy after "
                        f"{self._settings.startup_timeout:.0f}s"
                    )
                await asyncio.sleep(delay)
                delay = min(delay * 1.5, 2.0)

    def _write_state(self, engine: EngineProcess) -> None:
        self._settings.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._settings.state_path.write_text(json.dumps(self.info()))

    def _clear_state(self) -> None:
        self._settings.state_path.unlink(missing_ok=True)


def kill_recorded_engine(settings: Settings) -> int | None:
    """Force-kill the engine recorded in the state file (used when no server is up)."""
    path = settings.state_path
    try:
        pid = int(json.loads(path.read_text())["pid"])
    except (OSError, ValueError, KeyError):
        return None
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pid = None
    except PermissionError:
        return None
    path.unlink(missing_ok=True)
    return pid
