"""`coliollama scan`: show the hardware and which models it can reasonably run."""

from __future__ import annotations

import json
from typing import Annotated

import typer

from coliollama.cli.commands._shared import human_size
from coliollama.core import hardware
from coliollama.core.config import Settings
from coliollama.core.engine import gpu as gpulib
from coliollama.core.engine.installer import EngineInstaller
from coliollama.registry import catalog
from coliollama.registry.local_store import LocalStore

_RANK = {"fast": 0, "streamed": 1, "no-disk": 2}
_LABEL = {"fast": "fits", "streamed": "streams", "no-disk": "no disk"}


def scan(
    online: Annotated[bool, typer.Option("--online", help="Also query Hugging Face for more supported models.")] = False,
    all_models: Annotated[bool, typer.Option("--all", help="Include models that do not fit the disk.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Scan CPU, RAM, disk and GPUs, then rate every known downloadable model."""
    settings = Settings()
    hw = hardware.detect(settings.models_dir)
    installer = EngineInstaller(settings)
    directory = installer.current_dir()
    version = gpulib.read_version(directory)
    cuda_engine = bool(directory and gpulib.is_gpu_binary(directory / "colibri"))

    models = list(catalog.CATALOG)
    if online:
        models += catalog.discover_online()
    verdicts = sorted(
        (catalog.assess(m, hw, cuda_engine) for m in models),
        key=lambda v: (_RANK[v.level], v.model.converted_bytes),
    )
    installed = {e.repo_id for e in LocalStore(settings.registry_path).list() if e.repo_id}

    if as_json:
        typer.echo(json.dumps({
            "hardware": hw.to_dict(),
            "engine": {"version": version, "cuda": cuda_engine},
            "models": [
                {"repo": v.model.repo, "family": v.model.family, "verdict": v.level, "processor": v.processor,
                 "download_bytes": int(v.model.source_bytes), "converted_bytes": int(v.model.converted_bytes),
                 "peak_disk_bytes": int(v.disk_needed), "installed": v.model.repo in installed,
                 "verified": v.model.verified, "detail": v.detail or v.model.note}
                for v in verdicts
            ],
        }, indent=2))
        return

    typer.echo("Hardware")
    typer.echo(f"  CPU      {hw.cpu_threads} threads")
    typer.echo(f"  RAM      {human_size(hw.ram_total)} total, {human_size(hw.ram_available)} available")
    typer.echo(f"  Disk     {human_size(hw.disk_free)} free at {hw.disk_path}")
    for i, g in enumerate(hw.gpus):
        typer.echo(f"  GPU {i}    {g.name}, {human_size(g.vram_total)} VRAM ({human_size(g.vram_free)} free)")
    if not hw.gpus:
        typer.echo("  GPU      none detected")
    engine = f"Colibrí {version}" if version else "Colibrí not found (run `coliollama update`)"
    typer.echo(f"  Engine   {engine}{' (CUDA build)' if cuda_engine else ' (CPU build)' if version else ''}")
    if hw.gpus and version and not cuda_engine:
        typer.echo("           GPU* below = GPU-capable model, but run `coliollama update` to build the CUDA engine")

    typer.echo("\nModels")
    rows = [("MODEL", "FAMILY", "DOWNLOAD", "CONVERTED", "PEAK DISK", "RUNS ON", "STATUS")]
    hidden = 0
    for v in verdicts:
        if v.level == "no-disk" and not all_models:
            hidden += 1
            continue
        name = v.model.repo + ("" if v.model.verified else " (unverified)")
        status = "installed" if v.model.repo in installed else _LABEL[v.level]
        rows.append((name, v.model.family, human_size(v.model.source_bytes), human_size(v.model.converted_bytes),
                     human_size(v.disk_needed), v.processor, status))
    widths = [max(len(r[i]) for r in rows) for i in range(7)]
    for row in rows:
        typer.echo("  " + "  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip())
    typer.echo("\nfits    = converted model fits in RAM       streams = experts are read from disk (slower)")
    if hidden:
        typer.echo(f"{hidden} model(s) hidden because they do not fit the disk; use --all to list them.")
    notes = [f"  {v.model.repo}: {v.detail or v.model.note}" for v in verdicts
             if (v.detail or v.model.note) and v.level != "no-disk"]
    if notes:
        typer.echo("\nNotes\n" + "\n".join(notes))
    typer.echo("\nPull with: coliollama pull <MODEL>")
