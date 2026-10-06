"""Ollama-native endpoints: /api/tags, /api/ps, /api/chat, /api/generate (+ /api/stop)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from collections.abc import AsyncIterator
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from coliollama.api.fastapi.dependencies import (
    ApiError,
    acquire_lease,
    get_engine_client,
    get_scheduler,
    get_store,
    guarded_stream,
    json_body,
    lookup_model,
    prime,
)
from coliollama.core.engine.client import EngineClient, EngineHTTPError
from coliollama.core.scheduler.queue_manager import QueueManager
from coliollama.registry.local_store import LocalStore, ModelEntry

router = APIRouter(prefix="/api")

_OPTION_MAP = {
    "temperature": "temperature",
    "top_p": "top_p",
    "num_predict": "max_tokens",
    "stop": "stop",
    "seed": "seed",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
}

_TOOL_MODEL_TYPES = {
    "glm5_next", "glm5_next_text", "glm_moe_dsa", "glm5_moe", "glm",
    "kimi_k3", "kimi_linear", "qwen4_exp", "qwen4_exp_text",
    "deepseek_v4", "deepseek_v41", "deepseek_v41_text",
}
_VISION_MODEL_TYPES = {"glm5_next", "qwen4_exp", "deepseek_v41"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(entry: ModelEntry) -> str:
    return hashlib.sha256(entry.path.encode()).hexdigest()


def read_config(entry: ModelEntry) -> dict:
    try:
        config = json.loads((Path(entry.path) / "config.json").read_text())
    except (OSError, ValueError):
        return {}
    return config if isinstance(config, dict) else {}


def _int(config: dict, *keys: str) -> int | None:
    text = config.get("text_config") or config
    for key in keys:
        if isinstance(text.get(key), int):
            return text[key]
    return None


def context_length(entry: ModelEntry) -> int:
    return _int(read_config(entry), "max_position_embeddings") or 8192


def model_capabilities(entry: ModelEntry) -> list[str]:
    config = read_config(entry)
    text_config = config.get("text_config")
    model_types = {config.get("model_type")}
    if isinstance(text_config, dict):
        model_types.add(text_config.get("model_type"))
    capabilities = ["completion"]
    if model_types & _TOOL_MODEL_TYPES:
        capabilities.append("tools")
    if model_types & _VISION_MODEL_TYPES:
        capabilities.append("vision")
    return capabilities


def _details(entry: ModelEntry) -> dict:
    config = read_config(entry)
    family = config.get("model_type") or "moe"
    details = {"parent_model": entry.repo_id or "", "format": "colibri", "family": family,
               "families": [family], "parameter_size": "unknown", "quantization_level": "unknown",
               "context_length": context_length(entry)}
    embedding = _int(config, "hidden_size")
    if embedding:
        details["embedding_length"] = embedding
    return details


def _openai_params(body: dict) -> dict:
    params = {}
    for key, value in (body.get("options") or {}).items():
        if key in _OPTION_MAP and value is not None:
            params[_OPTION_MAP[key]] = value
    return params


def _timing(started: float, count: int, prompt_count: int | None) -> dict:
    out = {"total_duration": int((time.monotonic() - started) * 1e9), "eval_count": count}
    if prompt_count is not None:
        out["prompt_eval_count"] = prompt_count
    return out


@router.api_route("/tags", methods=["GET", "HEAD"])
async def tags(store: LocalStore = Depends(get_store)):
    models = []
    for m in store.list():
        modified = datetime.fromtimestamp(m.added_at or 0, timezone.utc).isoformat()
        models.append(
            {
                "name": m.name,
                "model": m.name,
                "modified_at": modified,
                "size": m.size_bytes() if m.exists else 0,
                "digest": _digest(m),
                "details": _details(m),
                "capabilities": model_capabilities(m),
            }
        )
    return {"models": models}


@router.get("/ps")
async def ps(request: Request, scheduler: QueueManager = Depends(get_scheduler), store: LocalStore = Depends(get_store)):
    snap = scheduler.snapshot()
    info = request.app.state.lifecycle.info()
    models = []
    if info:
        entry = store.get(info["model"])
        models.append(
            {
                "name": info["model"],
                "model": info["model"],
                "size": entry.size_bytes() if entry and entry.exists else 0,
                "digest": _digest(entry) if entry else "",
                "details": _details(entry) if entry else {},
                "expires_at": "0001-01-01T00:00:00Z",
                "size_vram": 0,
                "context_length": context_length(entry) if entry else 0,
                "pid": info["pid"],
                "processor": info["processor"],
                "started_at": datetime.fromtimestamp(info["started_at"], timezone.utc).isoformat(),
            }
        )
    return {"models": models, "queue": snap}


@router.get("/status")
async def status():
    return {"cloud": {"disabled": True, "source": "none"}}


@router.post("/stop")
async def stop(scheduler: QueueManager = Depends(get_scheduler)):
    return {"stopped": await scheduler.stop_engine()}


async def _load_only(model: str, scheduler: QueueManager, entry: ModelEntry, started: float, done_key: dict):
    lease = await acquire_lease(scheduler, entry)
    lease.release()
    return JSONResponse(
        {"model": model, "created_at": _now(), **done_key, "done": True, "done_reason": "load",
         "total_duration": int((time.monotonic() - started) * 1e9)}
    )


async def _respond(
    *,
    model: str,
    entry: ModelEntry,
    path: str,
    payload: dict,
    stream: bool,
    scheduler: QueueManager,
    client: EngineClient,
    make_chunk,
    empty_chunk: dict,
):
    """Run one engine request and translate OpenAI output to Ollama NDJSON / JSON.

    `make_chunk(text)` builds the content-bearing part of a response object.
    """
    started = time.monotonic()
    lease = await acquire_lease(scheduler, entry)

    if not stream:
        try:
            data = await client.post_json(lease.base_url, path, payload)
        except EngineHTTPError as exc:
            raise ApiError(exc.status, exc.message) from exc
        finally:
            lease.release()
        choice = (data.get("choices") or [{}])[0]
        text = choice.get("text") if "text" in choice else (choice.get("message") or {}).get("content", "")
        usage = data.get("usage") or {}
        return JSONResponse(
            {
                "model": model,
                "created_at": _now(),
                **make_chunk(text or ""),
                "done": True,
                "done_reason": choice.get("finish_reason") or "stop",
                **_timing(started, usage.get("completion_tokens", 0), usage.get("prompt_tokens")),
            }
        )

    try:
        source = await prime(client.stream_sse(lease.base_url, path, payload))
    except BaseException:
        lease.release()
        raise

    async def ndjson() -> AsyncIterator[str]:
        count = 0
        reason = "stop"
        usage: dict = {}
        async for raw in guarded_stream(lease, source):
            if raw == "[DONE]":
                break
            try:
                chunk = json.loads(raw)
            except ValueError:
                continue
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices") or []:
                text = choice.get("text") or (choice.get("delta") or {}).get("content") or ""
                if text:
                    count += 1
                    yield json.dumps({"model": model, "created_at": _now(), **make_chunk(text), "done": False}) + "\n"
                if choice.get("finish_reason"):
                    reason = choice["finish_reason"]
        yield json.dumps(
            {
                "model": model,
                "created_at": _now(),
                **empty_chunk,
                "done": True,
                "done_reason": reason,
                **_timing(started, usage.get("completion_tokens", count), usage.get("prompt_tokens")),
            }
        ) + "\n"

    return StreamingResponse(ndjson(), media_type="application/x-ndjson")


@router.post("/chat")
async def chat(
    body: dict = Depends(json_body),
    store: LocalStore = Depends(get_store),
    scheduler: QueueManager = Depends(get_scheduler),
    client: EngineClient = Depends(get_engine_client),
):
    entry = lookup_model(store, body.get("model"))
    messages = body.get("messages") or []
    if not messages:
        return await _load_only(entry.name, scheduler, entry, time.monotonic(), {"message": {"role": "assistant", "content": ""}})
    stream = body.get("stream", True)
    payload = {
        "model": entry.name,
        "messages": [{"role": m.get("role"), "content": m.get("content", "")} for m in messages],
        "stream": stream,
        **_openai_params(body),
    }
    return await _respond(
        model=entry.name, entry=entry, path="/v1/chat/completions", payload=payload,
        stream=stream, scheduler=scheduler, client=client,
        make_chunk=lambda text: {"message": {"role": "assistant", "content": text}},
        empty_chunk={"message": {"role": "assistant", "content": ""}},
    )


@router.post("/generate")
async def generate(
    body: dict = Depends(json_body),
    store: LocalStore = Depends(get_store),
    scheduler: QueueManager = Depends(get_scheduler),
    client: EngineClient = Depends(get_engine_client),
):
    entry = lookup_model(store, body.get("model"))
    prompt = body.get("prompt") or ""
    if not prompt:
        return await _load_only(entry.name, scheduler, entry, time.monotonic(), {"response": ""})
    stream = body.get("stream", True)
    if body.get("raw"):
        path = "/v1/completions"
        payload = {"model": entry.name, "prompt": prompt, "stream": stream, **_openai_params(body)}
    else:
        path = "/v1/chat/completions"
        messages = []
        if body.get("system"):
            messages.append({"role": "system", "content": body["system"]})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": entry.name, "messages": messages, "stream": stream, **_openai_params(body)}
    return await _respond(
        model=entry.name, entry=entry, path=path, payload=payload,
        stream=stream, scheduler=scheduler, client=client,
        make_chunk=lambda text: {"response": text}, empty_chunk={"response": ""},
    )
