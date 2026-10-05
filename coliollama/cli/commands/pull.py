"""`coliollama pull`: download from Hugging Face and register."""

from __future__ import annotations

from typing import Annotated

import typer

from coliollama.cli.commands._shared import human_path, human_size, resolve_or_exit


def pull(
    model: Annotated[str, typer.Argument(help="Hugging Face repo ID (org/repo), registry name or local directory.")],
    revision: Annotated[str | None, typer.Option(help="Branch, tag or commit to download.")] = None,
    no_convert: Annotated[bool, typer.Option("--no-convert", help="Skip the readiness check and conversion.")] = False,
    keep_source: Annotated[bool, typer.Option("--keep-source", help="Keep the raw download after converting.")] = False,
) -> None:
    """Fetch model weights from Hugging Face and register them locally."""
    entry = resolve_or_exit(model, revision=revision, convert=not no_convert, keep_source=keep_source)
    typer.secho(f"{entry.name}  {human_size(entry.size_bytes())}  {human_path(entry.path)}", fg=typer.colors.GREEN)
