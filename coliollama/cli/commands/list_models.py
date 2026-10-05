"""`coliollama list`: local model inventory."""

from __future__ import annotations

import typer

from coliollama.cli.commands._shared import human_path, human_size
from coliollama.core.config import Settings
from coliollama.registry.local_store import LocalStore


def list_models() -> None:
    """List local models with their sizes and paths."""
    models = LocalStore(Settings().registry_path).list()
    if not models:
        typer.echo("No models registered. Try `coliollama pull <org/repo>`.")
        return
    rows = [
        (m.name, human_size(m.size_bytes()) if m.exists else "missing", human_path(m.path))
        for m in models
    ]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(("NAME", "SIZE", "PATH"))]
    for row in [("NAME", "SIZE", "PATH"), *rows]:
        typer.echo("  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip())
