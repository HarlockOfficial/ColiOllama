"""`coliollama run`: resolve the model, make sure a server is up, chat."""

from __future__ import annotations

import json
import os
from typing import Annotated, Optional

import httpx
import typer

from coliollama.cli.commands._shared import (
    SERVER_OPTION_HELP,
    resolve_or_exit,
    server_up,
    server_url,
    spawn_server,
)


def _stream_reply(url: str, model: str, messages: list[dict]) -> str:
    reply = []
    with httpx.stream(
        "POST", f"{url}/api/chat",
        json={"model": model, "messages": messages, "stream": True},
        timeout=httpx.Timeout(None, connect=10.0),
    ) as resp:
        if resp.status_code >= 400:
            resp.read()
            try:
                msg = resp.json()["error"]
            except (ValueError, KeyError):
                msg = resp.text
            raise RuntimeError(msg)
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            text = chunk.get("message", {}).get("content", "")
            if text:
                reply.append(text)
                typer.echo(text, nl=False)
    typer.echo()
    return "".join(reply)


def run(
    model: Annotated[str, typer.Argument(help="Registry name, local directory or Hugging Face repo ID.")],
    prompt: Annotated[Optional[str], typer.Argument(help="One-shot prompt; omit for interactive chat.")] = None,
    host: Annotated[str | None, typer.Option(help=SERVER_OPTION_HELP)] = None,
    revision: Annotated[str | None, typer.Option(help="HF revision used if a download is needed.")] = None,
    no_convert: Annotated[bool, typer.Option("--no-convert", help="Skip the readiness check and conversion.")] = False,
    keep_source: Annotated[bool, typer.Option("--keep-source", help="Keep the raw download after converting.")] = False,
) -> None:
    """Chat with a model, downloading it from Hugging Face if it is missing."""
    entry = resolve_or_exit(model, revision=revision, convert=not no_convert, keep_source=keep_source)
    url = server_url(host)
    if not server_up(url):
        if host or os.environ.get("COLIOLLAMA_HOST"):
            typer.secho(f"No server at {url}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
        typer.secho("Starting ColiOllama server ...", fg=typer.colors.CYAN, err=True)
        url = spawn_server("127.0.0.1", 11434)

    history: list[dict] = []
    typer.secho(f"Model {entry.name} (first request loads the engine; this can take a while)", fg=typer.colors.CYAN, err=True)
    one_shot = prompt is not None
    while True:
        if one_shot:
            text = prompt
        else:
            try:
                text = typer.prompt(">>>", prompt_suffix=" ", default="", show_default=False).strip()
            except (EOFError, typer.Abort):
                typer.echo()
                return
        if not text:
            if one_shot:
                return
            continue
        if text in ("/bye", "/exit", "/quit"):
            return
        if text == "/clear":
            history.clear()
            typer.echo("Cleared session context")
            continue
        history.append({"role": "user", "content": text})
        try:
            answer = _stream_reply(url, entry.name, history)
        except (RuntimeError, httpx.HTTPError) as exc:
            history.pop()
            typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
            if one_shot:
                raise typer.Exit(1)
            continue
        history.append({"role": "assistant", "content": answer})
        if one_shot:
            return
