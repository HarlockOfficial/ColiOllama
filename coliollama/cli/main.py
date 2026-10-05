"""Root Typer application."""

from __future__ import annotations

import typer

from coliollama.cli.commands import list_models, process_status, pull, run, scan, serve, update

app = typer.Typer(
    name="coliollama",
    help="Ollama-style CLI and API for the Colibrí disk-streamed MoE engine.",
    no_args_is_help=True,
    add_completion=False,
)

app.command("serve")(serve.serve)
app.command("run")(run.run)
app.command("pull")(pull.pull)
app.command("list")(list_models.list_models)
app.command("ps")(process_status.ps)
app.command("scan")(scan.scan)
app.command("update")(update.update)
app.command("stop")(process_status.stop)

if __name__ == "__main__":
    app()
