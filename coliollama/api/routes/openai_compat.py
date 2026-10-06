"""OpenAI-compatible endpoints; requests are proxied to the engine through the scheduler."""

from __future__ import annotations

import json
import time
import uuid

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


def _response_input(value) -> list[dict]:
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if not isinstance(value, list):
        raise ApiError(400, "`input` must be a string or an array.", "invalid_request_error")

    messages = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ApiError(400, f"`input[{index}]` must be an object.", "invalid_request_error")
        kind = item.get("type", "message")
        if kind == "message":
            role = item.get("role")
            if role not in {"system", "developer", "user", "assistant"}:
                raise ApiError(400, f"`input[{index}].role` is not supported.", "invalid_request_error")
            content = item.get("content", "")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if not isinstance(part, dict):
                        raise ApiError(400, f"`input[{index}].content` parts must be objects.", "invalid_request_error")
                    part_type = part.get("type")
                    if part_type in {"input_text", "output_text", "text"}:
                        parts.append({"type": "text", "text": part.get("text", "")})
                    elif part_type == "input_image":
                        image = part.get("image_url")
                        parts.append({"type": "image_url", "image_url": image if isinstance(image, dict) else {"url": image}})
                    else:
                        parts.append(part)
                content = parts
            messages.append({"role": role, "content": content})
        elif kind == "function_call":
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": item.get("call_id") or item.get("id"),
                    "type": "function",
                    "function": {"name": item.get("name"), "arguments": item.get("arguments", "{}")},
                }],
            })
        elif kind == "function_call_output":
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ApiError(400, f"`input[{index}].call_id` is required.", "invalid_request_error")
            output = item.get("output", "")
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
            messages.append({"role": "tool", "tool_call_id": call_id, "content": output})
        else:
            raise ApiError(400, f"`input[{index}].type` '{kind}' is not supported.", "invalid_request_error")
    return messages


def _response_tools(tools) -> list[dict]:
    if not isinstance(tools, list):
        raise ApiError(400, "`tools` must be an array.", "invalid_request_error")
    converted = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ApiError(400, f"`tools[{index}]` must be a function tool.", "invalid_request_error")
        function = {key: tool[key] for key in ("name", "description", "parameters", "strict") if key in tool}
        if not isinstance(function.get("name"), str) or not function["name"]:
            raise ApiError(400, f"`tools[{index}].name` is required.", "invalid_request_error")
        converted.append({"type": "function", "function": function})
    return converted


def _responses_request(body: dict) -> dict:
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise ApiError(400, "`model` is required.", "invalid_request_error")
    messages = _response_input(body.get("input", ""))
    instructions = body.get("instructions")
    if instructions is not None:
        if not isinstance(instructions, str):
            raise ApiError(400, "`instructions` must be a string.", "invalid_request_error")
        messages.insert(0, {"role": "system", "content": instructions})
    request = {"model": model, "messages": messages, "stream": body.get("stream", False)}
    for source in (
        "temperature", "top_p", "tool_choice", "parallel_tool_calls",
        "stop", "seed", "presence_penalty", "frequency_penalty", "stream_options", "user",
    ):
        if source in body:
            request[source] = body[source]
    if "tools" in body:
        request["tools"] = _response_tools(body["tools"])
    tool_choice = request.get("tool_choice")
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        name = tool_choice.get("name")
        if not isinstance(name, str) or not name:
            raise ApiError(400, "`tool_choice.name` is required for a function choice.", "invalid_request_error")
        request["tool_choice"] = {"type": "function", "function": {"name": name}}
    if body.get("previous_response_id"):
        raise ApiError(400, "previous_response_id is not supported; include prior turns in `input`.",
                       "invalid_request_error")
    if body.get("background"):
        raise ApiError(400, "background responses are not supported.", "invalid_request_error")
    if "max_output_tokens" in body:
        request["max_tokens"] = body["max_output_tokens"]
    reasoning = body.get("reasoning")
    if reasoning is not None:
        if not isinstance(reasoning, dict):
            raise ApiError(400, "`reasoning` must be an object.", "invalid_request_error")
        if "effort" in reasoning:
            request["reasoning_effort"] = reasoning["effort"]
    text = body.get("text")
    if text is not None:
        if not isinstance(text, dict):
            raise ApiError(400, "`text` must be an object.", "invalid_request_error")
        fmt = text.get("format")
        if fmt is not None:
            if not isinstance(fmt, dict):
                raise ApiError(400, "`text.format` must be an object.", "invalid_request_error")
            if fmt.get("type") == "json_object":
                request["response_format"] = {"type": "json_object"}
            elif fmt.get("type") == "json_schema":
                request["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {key: fmt[key] for key in ("name", "description", "schema", "strict") if key in fmt},
                }
            elif fmt.get("type") != "text":
                raise ApiError(400, "`text.format.type` must be text, json_object, or json_schema.",
                               "invalid_request_error")
    return request


def _response_object(body: dict, completion: dict, response_id: str, created_at: int) -> dict:
    choices = completion.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    content = message.get("content") or ""
    output = []
    if content:
        output.append({
            "id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message", "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content, "annotations": []}],
        })
    for tool in message.get("tool_calls") or []:
        function = tool.get("function") or {}
        output.append({
            "id": tool.get("id") or f"fc_{uuid.uuid4().hex[:24]}", "type": "function_call",
            "status": "completed", "call_id": tool.get("id"), "name": function.get("name"),
            "arguments": function.get("arguments") or "{}",
        })
    usage = completion.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    incomplete = choice.get("finish_reason") == "length"
    return {
        "id": response_id, "object": "response", "created_at": created_at,
        "status": "incomplete" if incomplete else "completed", "error": None,
        "incomplete_details": {"reason": "max_output_tokens"} if incomplete else None,
        "instructions": body.get("instructions"),
        "max_output_tokens": body.get("max_output_tokens"),
        "model": body["model"], "output": output,
        "parallel_tool_calls": body.get("parallel_tool_calls", True),
        "temperature": body.get("temperature"), "tool_choice": body.get("tool_choice", "auto"),
        "tools": body.get("tools", []), "top_p": body.get("top_p"),
        "truncation": body.get("truncation", "disabled"),
        "metadata": body.get("metadata") or {},
        "usage": {
            "input_tokens": prompt_tokens, "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": completion_tokens, "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": usage.get("total_tokens", prompt_tokens + completion_tokens),
        },
    }


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


@router.post("/responses")
async def responses(
    body: dict = Depends(json_body),
    store: LocalStore = Depends(get_store),
    scheduler: QueueManager = Depends(get_scheduler),
    client: EngineClient = Depends(get_engine_client),
):
    request = _responses_request(body)
    if not isinstance(request["stream"], bool):
        raise ApiError(400, "`stream` must be a boolean.", "invalid_request_error")
    response_id = f"resp_{uuid.uuid4().hex}"
    created_at = int(time.time())

    if not request["stream"]:
        response = await _proxy("/v1/chat/completions", request, store, scheduler, client)
        completion = json.loads(response.body)
        return JSONResponse(_response_object(body, completion, response_id, created_at))

    entry = lookup_model(store, request["model"])
    request["model"] = entry.name
    lease = await acquire_lease(scheduler, entry)
    try:
        source = await prime(client.stream_sse(lease.base_url, "/v1/chat/completions", request))
    except BaseException:
        lease.release()
        raise

    async def events():
        text_parts = []
        tools = {}
        usage = {}
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        part = {"type": "output_text", "text": "", "annotations": []}
        message = {
            "id": message_id, "type": "message", "status": "in_progress",
            "role": "assistant", "content": [],
        }

        def response_state(status, output=None, error=None):
            return {
                "id": response_id, "object": "response", "created_at": created_at,
                "status": status, "error": error, "incomplete_details": None,
                "instructions": body.get("instructions"),
                "max_output_tokens": body.get("max_output_tokens"),
                "model": entry.name, "output": output or [],
                "parallel_tool_calls": body.get("parallel_tool_calls", True),
                "temperature": body.get("temperature"),
                "tool_choice": body.get("tool_choice", "auto"),
                "tools": body.get("tools", []), "top_p": body.get("top_p"),
                "truncation": body.get("truncation", "disabled"), "usage": None,
            }

        def event(name, fields):
            data = json.dumps({"type": name, **fields}, ensure_ascii=False)
            return f"event: {name}\ndata: {data}\n\n"

        try:
            yield event("response.created", {"response": response_state("in_progress")})
            yield event("response.in_progress", {"response": response_state("in_progress")})
            yield event("response.output_item.added", {"output_index": 0, "item": message})
            yield event("response.content_part.added", {
                "item_id": message_id, "output_index": 0, "content_index": 0, "part": part,
            })
            async for raw in source:
                if raw == "[DONE]":
                    break
                try:
                    chunk = json.loads(raw)
                except ValueError:
                    continue
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    text = delta.get("content")
                    if text:
                        text_parts.append(text)
                        yield event("response.output_text.delta", {
                            "item_id": message_id, "output_index": 0,
                            "content_index": 0, "delta": text,
                        })
                    for tool in delta.get("tool_calls") or []:
                        index = tool.get("index", 0)
                        state = tools.setdefault(index, {"id": None, "name": "", "arguments": ""})
                        if tool.get("id"):
                            state["id"] = tool["id"]
                        function = tool.get("function") or {}
                        state["name"] += function.get("name") or ""
                        state["arguments"] += function.get("arguments") or ""
                    if choice.get("finish_reason") == "length":
                        usage["incomplete"] = True
            text = "".join(text_parts)
            output = []
            final_message = {
                **message, "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
            if text:
                yield event("response.output_text.done", {
                    "item_id": message_id, "output_index": 0, "content_index": 0, "text": text,
                })
            yield event("response.content_part.done", {
                "item_id": message_id, "output_index": 0, "content_index": 0,
                "part": final_message["content"][0],
            })
            yield event("response.output_item.done", {"output_index": 0, "item": final_message})
            output.append(final_message)
            for _, state in sorted(tools.items()):
                call_id = state["id"] or f"call_{uuid.uuid4().hex[:24]}"
                item = {
                    "id": f"fc_{uuid.uuid4().hex[:24]}", "type": "function_call",
                    "status": "completed", "call_id": call_id, "name": state["name"],
                    "arguments": state["arguments"],
                }
                index = len(output)
                yield event("response.output_item.added", {
                    "output_index": index,
                    "item": {**item, "status": "in_progress", "arguments": ""},
                })
                yield event("response.function_call_arguments.delta", {
                    "item_id": item["id"], "output_index": index, "delta": state["arguments"],
                })
                yield event("response.function_call_arguments.done", {
                    "item_id": item["id"], "output_index": index, "arguments": state["arguments"],
                })
                yield event("response.output_item.done", {"output_index": index, "item": item})
                output.append(item)
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            result = response_state("incomplete" if usage.get("incomplete") else "completed", output)
            if usage.get("incomplete"):
                result["incomplete_details"] = {"reason": "max_output_tokens"}
            result["usage"] = {
                "input_tokens": prompt_tokens, "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": completion_tokens, "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": usage.get("total_tokens", prompt_tokens + completion_tokens),
            }
            yield event("response.completed", {"response": result})
        except EngineHTTPError as exc:
            yield event("response.failed", {
                "response": response_state("failed", error={"code": "engine_error", "message": exc.message}),
            })
        finally:
            await source.aclose()
            lease.release()

    return StreamingResponse(events(), media_type="text/event-stream")


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
