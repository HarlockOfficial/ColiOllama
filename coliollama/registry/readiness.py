"""Check whether a model directory is directly usable by Colibrí, and convert it if not.

`coli doctor` only validates tensor layout, so a raw Hugging Face checkpoint passes it, and
the engine's `/health` is green before any weights are read. The only reliable test is to
start the engine on the directory and generate one token (a "smoke test").
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from coliollama.core.config import Settings
from coliollama.core.engine.process import EngineProcess, ModelTarget, find_free_port

_ENVIRONMENT_PREFIXES = ("engine.", "accelerator.", "memory.", "storage.", "placement.")


class ReadinessError(Exception):
    pass


@dataclass
class ReadinessReport:
    model_problems: list[str] = field(default_factory=list)  # fixable by conversion
    environment_problems: list[str] = field(default_factory=list)  # not the model's fault

    @property
    def usable(self) -> bool:
        return not self.model_problems


def parse_doctor_report(report: dict) -> ReadinessReport:
    """Split `coli doctor` failures into machine problems and (default) model problems.

    Unknown check ids count against the model so that a broken model is never waved through.
    """
    out = ReadinessReport()
    for check in report.get("checks", []):
        if check.get("status") != "fail":
            continue
        line = f"{check.get('id')}: {check.get('summary')}"
        is_env = str(check.get("id", "")).startswith(_ENVIRONMENT_PREFIXES)
        bucket = out.environment_problems if is_env else out.model_problems
        bucket.append(line)
    return out


class ModelChecker:
    """`coli doctor --deep --json`, then a one-token generation on a throwaway engine."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def check(self, path: str | Path) -> ReadinessReport:
        report = self._doctor(path)
        if report.usable:
            problem = self._smoke_test(path)
            if problem:
                report.model_problems.append(f"smoke test: {problem}")
        return report

    def _smoke_test(self, path: str | Path) -> str | None:
        """Return None if the engine loads the model and generates a token, else why not."""
        settings = self._settings
        command = settings.resolve_coli()
        name = Path(path).name
        engine = EngineProcess(
            command, ModelTarget(name, str(path)), find_free_port(),
            settings.log_dir / "check.log", settings.engine_args,
        )
        engine.start()
        try:
            deadline = time.monotonic() + settings.startup_timeout
            with httpx.Client(timeout=5.0) as client:
                while True:
                    if not engine.is_alive():
                        return f"engine exited during startup: {engine.log_tail(5)}"
                    try:
                        if client.get(f"{engine.base_url}/health").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() > deadline:
                        return "engine did not become healthy in time"
                    time.sleep(0.3)
            body = {"model": name, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
            try:
                resp = httpx.post(
                    f"{engine.base_url}/v1/chat/completions", json=body,
                    timeout=httpx.Timeout(settings.startup_timeout, connect=10.0),
                )
            except httpx.HTTPError as exc:
                return f"generation request failed: {exc}; {engine.log_tail(3)}"
            if resp.status_code != 200:
                return f"engine returned HTTP {resp.status_code}: {engine.log_tail(3)}"
            return None
        finally:
            engine.terminate(settings.shutdown_timeout)

    def _doctor(self, path: str | Path) -> ReadinessReport:
        command = self._settings.resolve_coli()
        if not command:
            raise ReadinessError(
                "Colibrí `coli` launcher not found (set COLIOLLAMA_COLI or COLIBRI_HOME), "
                "so the model cannot be checked; use --no-convert to skip the check"
            )
        try:
            proc = subprocess.run(
                [*command, "doctor", "--model", str(path), "--deep", "--json"],
                capture_output=True, text=True, timeout=900, stdin=subprocess.DEVNULL,
            )
            return parse_doctor_report(json.loads(proc.stdout))
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ReadinessError(f"`coli doctor` could not run: {exc}") from exc
        except ValueError as exc:
            raise ReadinessError(
                f"`coli doctor` produced no readable report: {(proc.stderr or proc.stdout)[-300:]!r}"
            ) from exc


class ModelConverter:
    """Runs `coli convert --repo <hf repo> --model <outdir>` (download + convert shard by shard)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def convert(self, repo_id: str, out_dir: Path) -> None:
        command = self._settings.resolve_coli()
        if not command:
            raise ReadinessError("Colibrí `coli` launcher not found; cannot convert")
        missing = [m for m in ("numpy", "safetensors", "torch") if importlib.util.find_spec(m) is None]
        if missing and command[0] == sys.executable:
            raise ReadinessError(
                f"model conversion needs {', '.join(missing)} in this Python environment: "
                f"pip install {' '.join(missing)}"
            )
        if out_dir.exists():
            # coli refuses to write into a directory holding a checkpoint; restart cleanly.
            shutil.rmtree(out_dir)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        cmd = [*command, "convert", "--repo", repo_id, "--model", str(out_dir), *self._settings.convert_args]
        # Inherit the terminal so coli's own progress output is shown.
        env = dict(os.environ)
        if self._settings.hf_token:
            env["HF_TOKEN"] = self._settings.hf_token
        rc = subprocess.call(cmd, stdin=subprocess.DEVNULL, env=env)
        if rc != 0:
            raise ReadinessError(f"`coli convert` failed with exit code {rc} (partial output in {out_dir})")
