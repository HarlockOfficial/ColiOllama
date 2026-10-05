"""Request-scoped access to the shared services and model/lease helpers."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Request

from coliollama.core.engine.client import EngineClient, EngineHTTPError
from coliollama.core.engine.process import EngineStartError, ModelTarget
from coliollama.core.scheduler.queue_manager import Lease, QueueManager
from coliollama.registry.local_store import LocalStore, ModelEntry


class ApiError(Exception):
    def __init__(self, status: int, message: str, kind: str = "api_error") -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.kind = kind


async def json_body(request: Request) -> dict:
    """Parse the body as JSON whatever the Content-Type (`curl -d` sends form-urlencoded)."""
    try:
        data = json.loads(await request.body())
    except ValueError as exc:
        raise ApiError(400, f"invalid JSON body: {exc}", "invalid_request_error") from exc
    if not isinstance(data, dict):
        raise ApiError(400, "request body must be a JSON object", "invalid_request_error")
    return data


def get_store(request: Request) -> LocalStore:
    return request.app.state.store


def get_scheduler(request: Request) -> QueueManager:
    return request.app.state.scheduler


def get_engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def lookup_model(store: LocalStore, name: str | None) -> ModelEntry:
    if not name:
        raise ApiError(400, "model is required", "invalid_request_error")
    entry = store.get(name)
    if entry is None or not entry.exists:
        raise ApiError(404, f"model '{name}' not found; run `coliollama pull {name}`", "not_found")
    return entry


async def acquire_lease(scheduler: QueueManager, entry: ModelEntry) -> Lease:
    """Queue behind the scheduler policy; engine boot failures become 503s."""
    try:
        return await scheduler.acquire(ModelTarget(entry.name, entry.path))
    except EngineStartError as exc:
        raise ApiError(503, str(exc), "engine_unavailable") from exc
    except RuntimeError as exc:
        raise ApiError(503, str(exc), "engine_unavailable") from exc


async def guarded_stream(lease: Lease, source: AsyncIterator[Any]) -> AsyncIterator[Any]:
    """Re-yield `source`, releasing the lease when the stream ends or the client leaves."""
    try:
        async for item in source:
            yield item
    except EngineHTTPError:
        return  # headers are already sent; just end the stream
    finally:
        await source.aclose()
        lease.release()


async def prime(source: AsyncIterator[Any]) -> AsyncIterator[Any]:
    """Pull the first item eagerly so upstream HTTP errors surface before headers go out."""
    try:
        first = await source.__anext__()
    except StopAsyncIteration:
        async def empty():
            return
            yield

        return empty()
    except EngineHTTPError as exc:
        raise ApiError(exc.status, exc.message) from exc

    async def chained():
        try:
            yield first
            async for item in source:
                yield item
        finally:
            await source.aclose()

    return chained()
