"""`coliollama ps` and `coliollama stop`."""

from __future__ import annotations

import json
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
        data = httpx.get(f"{url}/api/ps", timeout=5.0).json()
    except httpx.HTTPError:
        typer.secho(f"No ColiOllama server reachable at {url}", fg=typer.colors.RED, err=True)
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


def stop(host: Annotated[str | None, typer.Option(help=SERVER_OPTION_HELP)] = None) -> None:
    """Force-terminate the active Colibrí engine process."""
    url = server_url(host)
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
