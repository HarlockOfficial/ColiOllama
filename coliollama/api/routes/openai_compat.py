"""OpenAI-compatible endpoints; requests are proxied to the engine through the scheduler."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends
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
from coliollama.registry.local_store import LocalStore

router = APIRouter(prefix="/v1")


async def _proxy(
    path: str,
    body: dict,
    store: LocalStore,
    scheduler: QueueManager,
    client: EngineClient,
):
    entry = lookup_model(store, body.get("model"))
    body = {**body, "model": entry.name}
    lease = await acquire_lease(scheduler, entry)
    if not body.get("stream"):
        try:
            return JSONResponse(await client.post_json(lease.base_url, path, body))
        except EngineHTTPError as exc:
            raise ApiError(exc.status, exc.message) from exc
        finally:
            lease.release()
    try:
        source = await prime(client.stream_sse(lease.base_url, path, body))
    except BaseException:
        lease.release()
        raise

    async def events():
        async for payload in guarded_stream(lease, source):
            yield f"data: {payload}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@router.post("/chat/completions")
async def chat_completions(
    body: dict = Depends(json_body),
    store: LocalStore = Depends(get_store),
    scheduler: QueueManager = Depends(get_scheduler),
    client: EngineClient = Depends(get_engine_client),
):
    return await _proxy("/v1/chat/completions", body, store, scheduler, client)


@router.post("/completions")
async def completions(
    body: dict = Depends(json_body),
    store: LocalStore = Depends(get_store),
    scheduler: QueueManager = Depends(get_scheduler),
    client: EngineClient = Depends(get_engine_client),
):
    return await _proxy("/v1/completions", body, store, scheduler, client)


@router.get("/models")
async def list_models(store: LocalStore = Depends(get_store)):
    return {
        "object": "list",
        "data": [
            {"id": m.name, "object": "model", "created": int(m.added_at or time.time()), "owned_by": "coliollama"}
            for m in store.list()
        ],
    }


@router.get("/models/{model_id:path}")
async def get_model(model_id: str, store: LocalStore = Depends(get_store)):
    entry = store.get(model_id)
    if entry is None:
        raise ApiError(404, f"The model '{model_id}' does not exist", "not_found_error")
    return {"id": entry.name, "object": "model", "created": int(entry.added_at or time.time()), "owned_by": "coliollama"}


@router.post("/embeddings")
async def embeddings(body: dict = Depends(json_body)):
    raise ApiError(501, "embeddings are not supported by ColiOllama (Colibrí cannot provide them)", "not_implemented")
