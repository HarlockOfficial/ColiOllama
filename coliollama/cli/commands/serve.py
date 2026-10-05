"""`coliollama serve`: boot the API server."""

from __future__ import annotations

from typing import Annotated

import typer

from coliollama.cli.commands._shared import server_up, spawn_server
from coliollama.core.config import Settings, default_server_url


def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Bind port (Ollama default 11434).")] = 11434,
    max_concurrency: Annotated[
        int, typer.Option("--max-concurrency", min=1, help="Concurrent requests admitted to the running engine.")
    ] = 1,
    log_level: Annotated[str, typer.Option(help="Uvicorn log level.")] = "info",
    gpu: Annotated[
        str,
        typer.Option(help="GPU use: auto (default, if the engine build and hardware support it), none, or device list like 0,1."),
    ] = "auto",
    auto_update: Annotated[
        bool, typer.Option("--auto-update/--no-auto-update", help="Check for new Colibrí releases and install them.")
    ] = True,
    detach: Annotated[bool, typer.Option("--detach", "-d", help="Run in the background.")] = False,
) -> None:
    """Start the Ollama/OpenAI-compatible API server."""
    if detach:
        if server_up(default_server_url(host, port)):
            typer.echo(f"Server already running on {host}:{port}")
            return
        url = spawn_server(
            host, port,
            ("--max-concurrency", str(max_concurrency), "--gpu", gpu,
             "--auto-update" if auto_update else "--no-auto-update"),
        )
        typer.echo(f"ColiOllama server running at {url}")
        return

    import uvicorn

    from coliollama.api.fastapi.app import create_app

    settings = Settings(host=host, port=port, max_concurrency=max_concurrency, gpu=gpu)
    settings.auto_update = settings.auto_update and auto_update
    uvicorn.run(create_app(settings), host=host, port=port, log_level=log_level)
