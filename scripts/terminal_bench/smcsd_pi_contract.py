#!/usr/bin/env python3
"""Validate the streamed OpenAI tool-calling contract used by Pi.

The first turn requests automatic tool selection and validates the model's
streamed JSON arguments.  Named/required tool choice is intentionally avoided:
SGLang implements it with constrained decoding, which SM-CSD does not support.
The second turn sends the resulting assistant/tool messages back to the server
and checks that generation can continue with the complete agent history.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Any


def post_stream(base_url: str, payload: dict[str, Any], timeout: float) -> dict:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": "Bearer local",
            "Content-Type": "application/json",
        },
    )
    started = time.monotonic()
    first_token = None
    content: list[str] = []
    calls: dict[int, dict[str, str]] = defaultdict(
        lambda: {"id": "", "type": "function", "name": "", "arguments": ""}
    )
    usage = None
    finish_reason = None

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                if not raw_line.startswith(b"data: "):
                    continue
                data = raw_line[6:].strip()
                if data == b"[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        content.append(delta["content"])
                        first_token = first_token or time.monotonic()
                    for call_delta in delta.get("tool_calls") or []:
                        first_token = first_token or time.monotonic()
                        call = calls[int(call_delta.get("index", 0))]
                        call["id"] += call_delta.get("id") or ""
                        call["type"] = call_delta.get("type") or call["type"]
                        function = call_delta.get("function") or {}
                        call["name"] += function.get("name") or ""
                        call["arguments"] += function.get("arguments") or ""
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc

    ended = time.monotonic()
    return {
        "content": "".join(content),
        "tool_calls": [calls[index] for index in sorted(calls)],
        "usage": usage,
        "finish_reason": finish_reason,
        "latency_s": ended - started,
        "ttft_s": None if first_token is None else first_token - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    tool = {
        "type": "function",
        "function": {
            "name": "get_working_directory",
            "description": "Return the current working directory.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are a terminal agent. Use tools when instructed.",
        },
        {
            "role": "user",
            "content": "Call get_working_directory now. Do not guess the result.",
        },
    ]
    common = {
        "model": args.model,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "max_tokens": 128,
        "tools": [tool],
    }
    first = post_stream(
        args.base_url,
        {
            **common,
            "messages": messages,
            "tool_choice": "auto",
        },
        args.timeout,
    )
    if len(first["tool_calls"]) != 1:
        raise SystemExit(f"expected one tool call, received: {first}")
    call = first["tool_calls"][0]
    if call["name"] != "get_working_directory":
        raise SystemExit(f"unexpected tool call: {call}")
    try:
        json.loads(call["arguments"] or "{}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid streamed tool arguments: {call}") from exc

    tool_call_id = call["id"] or "call_smcsd_contract"
    messages.extend(
        [
            {
                "role": "assistant",
                "content": first["content"] or None,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": call["arguments"] or "{}",
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": tool_call_id, "content": "/workspace"},
            {
                "role": "user",
                "content": "Report the directory returned by the tool in one sentence.",
            },
        ]
    )
    second = post_stream(
        args.base_url,
        {**common, "messages": messages, "tool_choice": "none"},
        args.timeout,
    )
    if not second["content"].strip():
        raise SystemExit(f"second turn returned no content: {second}")

    print(json.dumps({"status": "ok", "first": first, "second": second}, indent=2))


if __name__ == "__main__":
    main()
