"""Ollama model-management endpoints: version, show, copy, delete, pull, plus explicit
"not supported" answers for the ones Colibrí cannot back (create, push, blobs, embeddings)."""

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from coliollama import __version__
from coliollama.api.fastapi.dependencies import ApiError, get_scheduler, get_store, json_body, lookup_model
from coliollama.core.scheduler.queue_manager import QueueManager
from coliollama.registry.huggingface_resolver import HuggingFaceResolver, ModelResolutionError
from coliollama.registry.local_store import LocalStore

router = APIRouter(prefix="/api")


def _model_name(body: dict) -> str:
    name = body.get("model") or body.get("name")
    if not name:
        raise ApiError(400, "model is required", "invalid_request_error")
    return name


@router.get("/version")
async def version():
    return {"version": __version__}


@router.post("/show")
async def show(body: dict = Depends(json_body), store: LocalStore = Depends(get_store)):
    entry = lookup_model(store, _model_name(body))
    config: dict = {}
    try:
        config = json.loads((Path(entry.path) / "config.json").read_text())
    except (OSError, ValueError):
        pass
    text = config.get("text_config") or config
    info = {"general.architecture": config.get("model_type", "unknown")}
    for src, dst in (
        ("num_hidden_layers", "block_count"), ("hidden_size", "embedding_length"),
        ("num_experts", "expert_count"), ("n_routed_experts", "expert_count"),
        ("max_position_embeddings", "context_length"), ("vocab_size", "vocab_size"),
    ):
        if isinstance(text.get(src), int):
            info[f"general.{dst}"] = text[src]
    return {
        "modelfile": f"# Colibrí model\nFROM {entry.path}\n",
        "parameters": "",
        "template": "",
        "details": {"format": "colibri", "family": config.get("model_type", "moe"),
                    "parent_model": entry.repo_id or "", "parameter_size": "", "quantization_level": ""},
        "model_info": info,
        "capabilities": ["completion"],
    }


@router.post("/copy")
async def copy(body: dict = Depends(json_body), store: LocalStore = Depends(get_store)):
    source, dest = body.get("source"), body.get("destination")
    if not source or not dest:
        raise ApiError(400, "source and destination are required", "invalid_request_error")
    entry = lookup_model(store, source)
    # An alias: both names point at the same weights, nothing is duplicated on disk.
    store.add(dest, entry.path, repo_id=entry.repo_id, verified=entry.verified)
    return Response(status_code=200)


@router.delete("/delete")
async def delete(
    request: Request, body: dict = Depends(json_body),
    store: LocalStore = Depends(get_store), scheduler: QueueManager = Depends(get_scheduler),
):
    name = _model_name(body)
    entry = store.get(name)
    if entry is None:
        raise ApiError(404, f"model '{name}' not found", "not_found")
    if request.app.state.lifecycle.active_model == entry.name:
        raise ApiError(409, f"model '{name}' is loaded; run `coliollama stop` first", "conflict")
    store.remove(entry.name)
    settings = request.app.state.settings
    still_used = any(e.path == entry.path for e in store.list())
    managed = Path(entry.path).resolve().is_relative_to(settings.models_dir.resolve())
    if managed and not still_used:
        await asyncio.to_thread(shutil.rmtree, entry.path, True)
    return Response(status_code=200)


@router.post("/pull")
async def pull(request: Request, body: dict = Depends(json_body), store: LocalStore = Depends(get_store)):
    name = _model_name(body)
    stream = body.get("stream", True)
    resolver = HuggingFaceResolver(store, request.app.state.settings)
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def say(message: str) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, {"status": message})

    async def work() -> None:
        try:
            await asyncio.to_thread(resolver.ensure, name, on_status=say)
            await queue.put({"status": "success"})
        except ModelResolutionError as exc:
            await queue.put({"error": str(exc)})
        except Exception as exc:  # surface any failure to the client instead of dropping the stream
            await queue.put({"error": f"pull failed: {exc}"})
        await queue.put(None)

    task = asyncio.create_task(work())

    async def events() -> AsyncIterator[str]:
        while (item := await queue.get()) is not None:
            yield json.dumps(item) + "\n"
        await task

    if stream:
        return StreamingResponse(events(), media_type="application/x-ndjson")
    last: dict = {}
    async for line in events():
        last = json.loads(line)
    if "error" in last:
        return JSONResponse(last, status_code=500)
    return last


def _unsupported(what: str) -> ApiError:
    return ApiError(501, f"{what} is not supported by ColiOllama (Colibrí cannot provide it)", "not_implemented")


@router.post("/create")
async def create(body: dict = Depends(json_body)):
    raise _unsupported("creating models from a Modelfile; use `coliollama pull` or POST /api/copy")


@router.post("/push")
async def push(body: dict = Depends(json_body)):
    raise _unsupported("pushing models")


@router.post("/embed")
@router.post("/embeddings")
async def embed(body: dict = Depends(json_body)):
    raise _unsupported("embeddings")


@router.head("/blobs/{digest}")
async def blob_exists(digest: str):
    return Response(status_code=404)


@router.post("/blobs/{digest}")
async def blob_push(digest: str):
    raise _unsupported("blob upload")
