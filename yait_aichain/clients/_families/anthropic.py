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
            return json.loads(t)
        return text
