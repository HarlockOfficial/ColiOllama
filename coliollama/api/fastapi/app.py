"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from coliollama import __version__
from coliollama.api.fastapi.dependencies import ApiError
from coliollama.api.routes import ollama_compat, ollama_models, openai_compat
from coliollama.core.config import Settings
from coliollama.core.engine.client import EngineClient
from coliollama.core.engine.lifecycle import EngineLifecycle
from coliollama.core.scheduler.policy import SchedulerPolicy
from coliollama.core.scheduler.queue_manager import QueueManager
from coliollama.registry.local_store import LocalStore


log = logging.getLogger("coliollama.update")


async def _auto_update_loop(settings: Settings) -> None:
    from coliollama.core.engine.installer import EngineInstaller, UpdateError

    installer = EngineInstaller(settings)
    while True:
        try:
            if await asyncio.to_thread(installer.due):
                await asyncio.to_thread(installer.update, log.info)
        except UpdateError as exc:
            log.warning("engine update skipped: %s", exc)
        except Exception:  # never let the checker take the server down
            log.exception("engine update failed")
        await asyncio.sleep(min(settings.update_interval, 3600))


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        lifecycle = EngineLifecycle(settings)
        app.state.settings = settings
        app.state.store = LocalStore(settings.registry_path)
        app.state.lifecycle = lifecycle
        app.state.scheduler = QueueManager(lifecycle, SchedulerPolicy(settings.max_concurrency))
        app.state.engine_client = EngineClient()
        updater = None
        if settings.auto_update and not settings.engine_pinned:
            if not log.handlers and not logging.getLogger().handlers:
                logging.basicConfig(level=logging.INFO, format="%(levelname)s:  %(message)s")
            updater = asyncio.create_task(_auto_update_loop(settings))
        try:
            yield
        finally:
            if updater:
                updater.cancel()
            await app.state.scheduler.shutdown()
            await app.state.engine_client.aclose()

    app = FastAPI(title="ColiOllama", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        if request.url.path.startswith("/v1/"):
            body = {"error": {"message": exc.message, "type": exc.kind, "param": None, "code": None}}
        else:
            body = {"error": exc.message}
        return JSONResponse(body, status_code=exc.status)

    @app.api_route("/", methods=["GET", "HEAD"])
    async def root() -> PlainTextResponse:
        return PlainTextResponse("ColiOllama is running")

    app.include_router(ollama_compat.router)
    app.include_router(ollama_models.router)
    app.include_router(openai_compat.router)
    return app
