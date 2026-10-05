"""`coliollama update`: check GitHub for a newer Colibrí and install it as the main engine."""

from __future__ import annotations

from typing import Annotated

import typer

from coliollama.core.config import Settings
from coliollama.core.engine.installer import EngineInstaller, UpdateError


def update(
    check: Annotated[bool, typer.Option("--check", help="Only report; install nothing.")] = False,
    force: Annotated[bool, typer.Option("--force", help="Reinstall the latest release even if current.")] = False,
    gpu_build: Annotated[
        bool | None,
        typer.Option("--gpu-build/--no-gpu-build", help="Build CUDA engines from source (default: when an NVIDIA GPU and nvcc exist)."),
    ] = None,
) -> None:
    """Install the latest Colibrí release (and its CUDA build when possible)."""
    settings = Settings()
    installer = EngineInstaller(settings)
    try:
        status, latest = installer.check()
        typer.echo(f"installed: {status.current or 'none'}   latest: {status.latest}")
        if status.pinned:
            typer.secho(
                "The engine is pinned by COLIOLLAMA_COLI/COLIBRI_HOME; unset it to let ColiOllama manage updates.",
                fg=typer.colors.YELLOW,
            )
            if not check:
                raise typer.Exit(1)
        if check:
            if status.update_available:
                typer.echo("A newer release is available; run `coliollama update`.")
            else:
                typer.echo("Colibrí is up to date.")
            if status.gpu_build_needed:
                typer.echo("An NVIDIA GPU and CUDA toolkit were found but the engine is CPU-only; "
                           "`coliollama update` will build the CUDA engines.")
            return
        installer.update(say=typer.echo, force=force, build_gpu=gpu_build, retry_gpu=gpu_build is True)
    except UpdateError as exc:
        typer.secho(f"update failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
