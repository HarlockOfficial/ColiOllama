"""Runtime settings, resolved from environment variables."""

from __future__ import annotations

import os
import shlex
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path


def load_dotenv_values(*paths: Path) -> dict[str, str]:
    """Parse simple KEY=VALUE .env files; earlier paths win. Real environment variables win over all."""
    values: dict[str, str] = {}
    for path in paths:
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip().removeprefix("export ").strip()
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                val = val[1:-1]
            values.setdefault(key, val)
    return values


def _default_hf_token() -> str | None:
    home = Path(os.environ.get("COLIOLLAMA_HOME", "~/.coliollama")).expanduser()
    dotenv = load_dotenv_values(Path.cwd() / ".env", home / ".env")
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(key) or dotenv.get(key)
        if token:
            return token
    return None


def _launcher(path: Path) -> tuple[str, ...]:
    # `coli` is a Python script; run it with our interpreter so the exec bit doesn't matter.
    return (sys.executable, str(path))


def _find_coli(engines_dir: Path) -> tuple[str, ...] | None:
    """COLIOLLAMA_COLI > COLIBRI_HOME > managed install > ./dependencies/colibri > project > PATH."""
    from coliollama.core.engine import installer

    explicit = os.environ.get("COLIOLLAMA_COLI")
    if explicit:
        return tuple(shlex.split(explicit))
    roots = []
    if os.environ.get("COLIBRI_HOME"):
        roots.append(Path(os.environ["COLIBRI_HOME"]).expanduser())
    else:
        managed = installer.managed_root(engines_dir)
        if managed:
            roots.append(managed)
    roots.append(Path.cwd() / "dependencies" / "colibri")
    roots.append(Path(__file__).resolve().parents[2] / "dependencies" / "colibri")
    for root in roots:
        for candidate in (root / "coli", root / "c" / "coli"):
            if candidate.is_file():
                return _launcher(candidate)
    found = shutil.which("coli")
    return (found,) if found else None


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() not in ("0", "false", "no", "off", "")


@dataclass
class Settings:
    home: Path = field(
        default_factory=lambda: Path(
            os.environ.get("COLIOLLAMA_HOME", "~/.coliollama")
        ).expanduser()
    )
    host: str = "127.0.0.1"
    port: int = 11434
    max_concurrency: int = 1
    startup_timeout: float = field(
        default_factory=lambda: float(os.environ.get("COLIOLLAMA_STARTUP_TIMEOUT", "900"))
    )
    shutdown_timeout: float = 20.0
    # Pin the launcher explicitly; None means "resolve it each time" so that an engine
    # update takes effect on the next engine start without restarting the server.
    coli_command: tuple[str, ...] | None = None
    gpu: str = field(default_factory=lambda: os.environ.get("COLIOLLAMA_GPU", "auto"))
    auto_update: bool = field(default_factory=lambda: env_flag("COLIOLLAMA_AUTO_UPDATE", True))
    update_interval: float = field(
        default_factory=lambda: float(os.environ.get("COLIOLLAMA_UPDATE_INTERVAL_HOURS", "24")) * 3600
    )
    engine_args: tuple[str, ...] = field(
        default_factory=lambda: tuple(shlex.split(os.environ.get("COLIOLLAMA_ENGINE_ARGS", "")))
    )

    # From HF_TOKEN in the environment, ./.env or <home>/.env (needed for gated repos such as meta-llama).
    hf_token: str | None = field(default_factory=_default_hf_token, repr=False)

    convert_args: tuple[str, ...] = field(
        default_factory=lambda: tuple(shlex.split(os.environ.get("COLIOLLAMA_CONVERT_ARGS", "")))
    )

    def resolve_coli(self) -> tuple[str, ...] | None:
        return self.coli_command or _find_coli(self.engines_dir)

    @property
    def engine_pinned(self) -> bool:
        """True when the user chose the engine explicitly, so it must not be auto-managed."""
        return bool(
            self.coli_command
            or os.environ.get("COLIOLLAMA_COLI")
            or os.environ.get("COLIBRI_HOME")
        )

    @property
    def engines_dir(self) -> Path:
        return self.home / "engines"

    @property
    def models_dir(self) -> Path:
        return self.home / "models"

    @property
    def registry_path(self) -> Path:
        return self.home / "registry.json"

    @property
    def state_path(self) -> Path:
        return self.home / "engine.json"

    def server_pid_path(self, port: int) -> Path:
        return self.home / "servers" / f"{port}.pid"

    @property
    def log_dir(self) -> Path:
        return self.home / "logs"

    @property
    def server_url(self) -> str:
        return default_server_url(self.host, self.port)


def default_server_url(host: str = "127.0.0.1", port: int = 11434) -> str:
    env = os.environ.get("COLIOLLAMA_HOST")
    if env:
        return env if "://" in env else f"http://{env}"
    host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    return f"http://{host}:{port}"
