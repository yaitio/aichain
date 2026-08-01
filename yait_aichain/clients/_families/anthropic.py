"""
clients._families.anthropic
============================

Anthropic Messages API family (``POST /v1/messages``): format
(build_request / parse_response) + transport (x-api-key + version header).

System messages lift to a top-level ``system`` field; structured output uses
a forced ``tool_use``; extended thinking maps to ``budget_tokens``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from .._base import BaseClient


def _extract_first_json(t: str) -> str:
    """Return the first balanced top-level JSON object/array in t.

    Models sometimes append commentary after the JSON, or wrap it in prose.
    Scans for the first { or [ and walks until the matching close, respecting
    string literals and escapes."""
    start = next((i for i, ch in enumerate(t) if ch in "{["), -1)
    if start == -1:
        raise json.JSONDecodeError("no JSON object found", t, 0)
    open_ch  = t[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:        esc = False
            elif ch == "\\": esc = True
            elif ch == '"':  in_str = False
        else:
            if ch == '"':       in_str = True
            elif ch == open_ch:  depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return t[start:i + 1]
    raise json.JSONDecodeError("unbalanced JSON", t, start)

_API_VERSION = "2023-06-01"


def _part_to_anthropic(part: dict) -> "dict | None":
    """
    Convert one universal part dict to an Anthropic content block.

    Returns ``None`` for unsupported types (video) so callers can filter.
    """
    ptype = part["type"]

    if ptype == "text":
        return {"type": "text", "text": part["text"]}

    if ptype == "image":
        src  = part["source"]
        kind = src["kind"]
        if kind == "url":
            return {
                "type":   "image",
                "source": {"type": "url", "url": src["url"]},
            }
        if kind in ("base64", "file"):
            return {
                "type":   "image",
                "source": {
                    "type":       "base64",
                    "media_type": src.get("mime", "image/png"),
                    "data":       src["data"],
                },
            }

    if ptype == "audio":
        src  = part["source"]
        kind = src["kind"]
        if kind == "base64":
            return {
                "type":   "document",
                "source": {
                    "type":       "base64",
                    "media_type": src.get("mime", "audio/wav"),
                    "data":       src["data"],
                },
            }
        return None

    # ptype == "video" — not supported by Anthropic
    return None


class AnthropicClient(BaseClient):

    def __init__(self, api_key: str, *, data: dict, **client_opts) -> None:
        prov = data["provider"]
        super().__init__(
            api_key,
            url=client_opts.get("url") or prov.get("base_url"),
            **{k: client_opts[k] for k in ("timeout", "retries", "proxy")
               if k in client_opts},
        )
        self._data = data

    # ── transport ────────────────────────────────────────────────────
    def _auth_headers(self) -> dict:
        return {"x-api-key": self._api_key,
                "anthropic-version": _API_VERSION,
                "Content-Type": "application/json"}

    def list_models(self) -> list[str]:
        data = self._get("/v1/models", self._auth_headers())
        return [m["id"] for m in json.loads(data)["data"]]

    # ── format ───────────────────────────────────────────────────────
    def build_request(self, messages, output, params) -> "tuple[str, dict]":
        prov = self._data["provider"]
        rmap = prov.get("reasoning_map", {})
        default_max = prov["defaults"]["max_tokens"]
        name        = params["name"]
        reasoning   = params.get("reasoning")

        system_parts: list[dict] = []
        amsgs: list[dict] = []
        for msg in messages:
            blocks = [_part_to_anthropic(p) for p in msg["parts"]]
            blocks = [b for b in blocks if b is not None]
            if not blocks:
                continue
            if msg["role"] == "system":
                system_parts.extend(blocks)
            else:
                amsgs.append({"role": msg["role"], "content": blocks})

        body: dict = {
            "model":       name,
            "messages":    amsgs,
            "max_tokens":  params["max_tokens"],
            "temperature": params["temperature"],
        }
        if system_parts:
            if all(b["type"] == "text" for b in system_parts):
                body["system"] = "\n\n".join(b["text"] for b in system_parts)
            else:
                body["system"] = system_parts
        # ── prompt cache breakpoint ──────────────────────────────────────
        # Everything up to and including the marked block is cached. The mark
        # goes on the last block of the second-to-last message, so the newest
        # turn stays outside it: on the next call that turn has become part of
        # the stable prefix and is read back rather than recomputed. Marking
        # the newest message instead would store a prefix that never repeats.
        # ``cache_control`` may be True, which marks the second-to-last message,
        # or an integer index naming the message the stable prefix ends at.
        # The default guess is only right when the array grows by appending;
        # a caller that rebuilds part of the array — a memory that sends
        # retrieved context plus a sliding window of recent turns — has a
        # prefix that ends well before the last message, and marking past it
        # asks the provider to cache content that changes every call. Measured:
        # a composed prompt whose first 29 of 36 messages were byte-identical
        # turn after turn got a cache read of zero, because the mark sat on
        # message 34.
        mark = params.get("cache_control")
        by_index = isinstance(mark, int) and not isinstance(mark, bool)
        if mark is True or by_index:
            target = None
            if by_index:
                # Absolute index only, and index 0 is a real answer: a memory
                # that puts a fixed preamble first is saying exactly that.
                # Testing the mark for truthiness threw it away, because zero
                # is false. Python would also read -1 as the *last* message,
                # the opposite of what a caller reporting "nothing is stable
                # yet" means by it. Both mistakes surface the same way — a
                # cache that silently never hits — which is why an index that
                # names no message marks nothing rather than being second-
                # guessed: an explicit index is an instruction, not a hint.
                if 0 <= mark < len(amsgs):
                    target = amsgs[mark]["content"]
            else:
                if len(amsgs) >= 2:
                    target = amsgs[-2]["content"]
                if target is None and system_parts:
                    # An all-text system prompt was collapsed into a string a
                    # few lines above, and a string has no block to carry the
                    # mark. Normalising it back is what makes
                    # ``cache_control=True`` mean anything for the commonest
                    # skill shape there is — one system prompt plus one short
                    # user message — where ``amsgs`` holds a single message and
                    # the branch above cannot fire. Marking that message
                    # instead would be worse than doing nothing: it asks the
                    # provider to store a prefix that changes every call.
                    if isinstance(body.get("system"), str):
                        body["system"] = [{"type": "text", "text": body["system"]}]
                    target = body["system"]
            if target:
                control = {"type": "ephemeral"}
                if params.get("cache_ttl") == "1h":
                    control["ttl"] = "1h"
                for blk in reversed(target):
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        blk["cache_control"] = control
                        break

        if params.get("top_p") is not None:
            body["top_p"] = params["top_p"]
        if params.get("top_k") is not None:
            body["top_k"] = params["top_k"]

        if reasoning:
            budget = rmap.get(reasoning)
            if budget is not None:
                body["thinking"]    = {"type": "enabled", "budget_tokens": budget}
                body["temperature"] = 1.0
                if body["max_tokens"] <= budget:
                    body["max_tokens"] = budget + default_max

        fmt = output.get("format", {})
        if fmt.get("type") == "json_schema":
            tool = fmt.get("name", "structured_output")
            body["tools"] = [{"name": tool,
                              "description": "Return the result matching the given schema.",
                              "input_schema": fmt["schema"]}]
            body["tool_choice"] = {"type": "tool", "name": tool}

        return "/v1/messages", body

    def parse_response(self, response, output) -> "str | dict":
        ftype   = output.get("format", {}).get("type", "text")
        content = response.get("content", [])
        if ftype == "json_schema":
            for block in content:
                if block.get("type") == "tool_use":
                    return block.get("input", {})
            return {}
        text = ""
        for block in content:
            if block.get("type") == "text":
                text = block["text"]; break
        if ftype == "json":
            t = text.strip()
            if t.startswith("```"):
                t = t.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            try:
                return json.loads(t)
            except json.JSONDecodeError:
                # Model emitted trailing prose after the JSON, or led with it.
                # Recover the first balanced top-level {...} / [...] object.
                return json.loads(_extract_first_json(t))
        return text
