"""`coliollama ps` and `coliollama stop`."""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path
from urllib.parse import urlparse
from typing import Annotated

import httpx
import typer

from coliollama.cli.commands._shared import SERVER_OPTION_HELP, server_url
from coliollama.core.config import Settings
from coliollama.core.engine.lifecycle import kill_recorded_engine


def ps(host: Annotated[str | None, typer.Option(help=SERVER_OPTION_HELP)] = None) -> None:
    """Show the running model, engine PID and queue depth."""
    url = server_url(host)
    try:
        resp = httpx.get(f"{url}/api/ps", timeout=5.0)
        data = resp.json()
    except (httpx.HTTPError, ValueError):
        typer.secho(f"No ColiOllama server reachable at {url}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    if not isinstance(data, dict) or "queue" not in data:
        detail = data.get("error") if isinstance(data, dict) else None
        typer.secho(
            f"{url} answered /api/ps but is not a ColiOllama server"
            + (f" ({detail})" if detail else "")
            + ". Another service (e.g. Ollama) may be using that port; "
            "start ColiOllama with `coliollama serve --port <port>` and pass `--host`.",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(1)
    queue = data["queue"]
    models = data["models"]
    rows = [("MODEL", "PID", "PROCESSOR", "QUEUE", "IN FLIGHT", "WAITING")]
    if models:
        m = models[0]
        rows.append((m["name"], str(m["pid"]), m.get("processor", "-").upper(), str(queue["queue_depth"]), str(queue["in_flight"]), str(queue["waiting"])))
    elif queue["loading_model"]:
        rows.append((f"{queue['loading_model']} (loading)", "-", "-", str(queue["queue_depth"]), "0", str(queue["waiting"])))
    else:
        typer.echo("No model loaded.")
        return
    widths = [max(len(r[i]) for r in rows) for i in range(6)]
    for row in rows:
        typer.echo("  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip())


def _is_our_server(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
    except OSError:
        return True  # no /proc (macOS): trust the pid file
    return "coliollama" in cmdline


def _stop_server(url: str) -> None:
    port = urlparse(url).port or 11434
    settings = Settings()
    pid_file = settings.server_pid_path(port)
    try:
        pid = int(pid_file.read_text())
    except (OSError, ValueError):
        typer.echo(f"No ColiOllama server recorded for port {port} (was it started with `serve`?)")
        raise typer.Exit(1)
    if not _is_our_server(pid):
        pid_file.unlink(missing_ok=True)
        typer.echo(f"Server on port {port} is not running (stale pid file removed).")
        return
    os.kill(pid, signal.SIGTERM)  # graceful: the server stops its engine on the way out
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and _is_our_server(pid):
        time.sleep(0.2)
    if _is_our_server(pid):
        os.kill(pid, signal.SIGKILL)
        kill_recorded_engine(settings)
        typer.echo(f"Server {pid} did not exit in time and was killed.")
    else:
        typer.echo(f"Server {pid} on port {port} stopped.")
    pid_file.unlink(missing_ok=True)


def stop(
    host: Annotated[str | None, typer.Option(help=SERVER_OPTION_HELP)] = None,
    server: Annotated[
        bool, typer.Option("--server", "-s", help="Stop the ColiOllama server itself (and its engine), not just the engine.")
    ] = False,
) -> None:
    """Force-terminate the active Colibrí engine; with --server, shut the server down too."""
    url = server_url(host)
    if server:
        _stop_server(url)
        return
    try:
        stopped = httpx.post(f"{url}/api/stop", timeout=60.0).json().get("stopped")
    except (httpx.HTTPError, json.JSONDecodeError):
        pid = kill_recorded_engine(Settings())
        if pid:
            typer.echo(f"Killed engine process group {pid}")
        else:
            typer.echo("No engine running.")
        return
    typer.echo("Engine stopped." if stopped else "No engine running.")
