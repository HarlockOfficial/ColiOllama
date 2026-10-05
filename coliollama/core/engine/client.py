"""HTTP client for the OpenAI-compatible gateway exposed by a running engine."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx


class EngineHTTPError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _error_message(status: int, body: bytes) -> str:
    import json

    try:
        err = json.loads(body).get("error")
        if isinstance(err, dict):
            return str(err.get("message", err))
        if err:
            return str(err)
    except (ValueError, AttributeError):
        pass
    return body.decode(errors="replace")[:500] or f"engine returned HTTP {status}"


class EngineClient:
    def __init__(self) -> None:
        # Generation on disk-streamed models can be very slow: no read timeout.
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10.0))

    async def post_json(self, base_url: str, path: str, body: dict) -> dict:
        try:
            resp = await self._http.post(base_url + path, json=body)
        except httpx.HTTPError as exc:
            raise EngineHTTPError(502, f"engine unreachable: {exc}") from exc
        if resp.status_code >= 400:
            raise EngineHTTPError(resp.status_code, _error_message(resp.status_code, resp.content))
        return resp.json()

    async def stream_sse(self, base_url: str, path: str, body: dict) -> AsyncIterator[str]:
        """Yield the payload of each SSE `data:` line (including `[DONE]`)."""
        try:
            async with self._http.stream("POST", base_url + path, json=body) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    raise EngineHTTPError(resp.status_code, _error_message(resp.status_code, resp.content))
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        yield line[5:].strip()
        except httpx.HTTPError as exc:
            raise EngineHTTPError(502, f"engine connection lost: {exc}") from exc

    async def aclose(self) -> None:
        await self._http.aclose()
