"""Helpers shared by CLI commands."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import httpx
import typer

from coliollama.core.config import Settings, default_server_url
from coliollama.registry.huggingface_resolver import HuggingFaceResolver, ModelResolutionError
from coliollama.registry.local_store import LocalStore, ModelEntry

SERVER_OPTION_HELP = "Server URL (env COLIOLLAMA_HOST)."


def server_url(override: str | None) -> str:
    if override:
        return override if "://" in override else f"http://{override}"
    return default_server_url()


def server_up(url: str) -> bool:
    try:
        return httpx.get(f"{url}/api/tags", timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


def resolve_or_exit(
    ref: str,
    *,
    revision: str | None = None,
    allow_download: bool = True,
    convert: bool = True,
    keep_source: bool = False,
) -> ModelEntry:
    settings = Settings()
    resolver = HuggingFaceResolver(LocalStore(settings.registry_path), settings)
    try:
        return resolver.ensure(
            ref,
            revision=revision,
            allow_download=allow_download,
            convert=convert,
            keep_source=keep_source,
            on_status=lambda msg: typer.secho(msg, fg=typer.colors.CYAN, err=True),
        )
    except ModelResolutionError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


def spawn_server(host: str, port: int, extra_args: tuple[str, ...] = (), wait: float = 30.0) -> str:
    """Start a detached `coliollama serve` and wait until it answers."""
    settings = Settings()
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    log = open(settings.log_dir / "server.log", "ab")
    try:
        subprocess.Popen(
            [sys.executable, "-m", "coliollama", "serve", "--host", host, "--port", str(port), *extra_args],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    finally:
        log.close()
    url = default_server_url(host, port)
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if server_up(url):
            return url
        time.sleep(0.3)
    typer.secho(f"Server did not start; see {settings.log_dir / 'server.log'}", fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


def human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024 or unit == "TB":
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024
    return str(num)


def human_path(path: str) -> str:
    try:
        return "~/" + str(Path(path).relative_to(Path.home()))
    except ValueError:
        return path
