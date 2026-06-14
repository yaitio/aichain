"""
clients._families.openai
========================

OpenAI Chat Completions API family — one client for the six OpenAI-compatible
providers (openai, xai, perplexity, kimi, deepseek, qwen).  Holds BOTH the
format (build_request / parse_response) and the Bearer transport.

The client is created **per provider** (it gets the provider's data dict), so
it knows its endpoints, base URL, and which quirk branches apply — while the
model-level settings (name, temperature, …) arrive in ``params`` on each call.

The wire-format helpers live alongside in ``_openai_compat`` (format is a
property of the API, so it belongs in the client layer).
"""

from __future__ import annotations

import json as _json
from types import SimpleNamespace

from .._base import BaseClient
from ._openai_compat import (
    _part_to_openai,                       # noqa: F401  (kept for parity)
    _build_openai_compat_request,
    _parse_openai_compat_response,
    _build_responses_api_request,
    _parse_responses_api_response,
    _build_image_generations_request,
    _parse_image_generations_response,
    _is_openai_image_model,
    _is_o_series_model,
    _should_use_responses_api,
)


# ── model-name gates (by prefix) ────────────────────────────────────────────
def _is_xai_image(name: str) -> bool:    return name.startswith("grok-imagine-")
def _is_qwen_image(name: str) -> bool:   return name.startswith("wanx")
def _is_qwq(name: str) -> bool:          return name.lower().startswith("qwq")
def _is_qwen3(name: str) -> bool:        return name.lower().startswith("qwen3")
def _is_deepseek_reasoner(name: str) -> bool: return "reasoner" in name


class OpenAIClient(BaseClient):
    """Bearer-auth client for the OpenAI Chat Completions family."""

    def __init__(self, api_key: str, *, data: dict, **client_opts) -> None:
        prov = data["provider"]
        super().__init__(
            api_key,
            url=client_opts.get("url") or prov.get("base_url"),
            **{k: client_opts[k] for k in ("timeout", "retries", "proxy")
               if k in client_opts},
        )
        self._data        = data
        self._provider    = prov["key"]
        self._chat_path   = prov.get("chat_path", "/v1/chat/completions")
        self._images_path = prov.get("images_path", "/v1/images/generations")
        self._models_path = prov.get("models_path", "/v1/models")
        self._mtf         = prov.get("max_tokens_field", "max_completion_tokens")

    # ── transport ────────────────────────────────────────────────────
    def _auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json"}

    def list_models(self) -> list[str]:
        data = self._get(self._models_path, self._auth_headers())
        return [m["id"] for m in _json.loads(data)["data"]]

    # ── format: model params → provider body ─────────────────────────
    def _wrap(self, params: dict):
        prov = self._data["provider"]
        return SimpleNamespace(
            name             = params["name"],
            temperature      = params.get("temperature"),
            max_tokens       = params.get("max_tokens"),
            top_p            = params.get("top_p"),
            top_k            = params.get("top_k"),
            reasoning        = params.get("reasoning"),
            _REASONING_MAP   = prov.get("reasoning_map", {}),
            _DEFAULT_MAX_TOKENS = prov["defaults"]["max_tokens"],
        )

    def build_request(self, messages, output, params) -> "tuple[str, dict]":
        m   = self._wrap(params)
        p   = self._provider

        if p == "openai":
            if _should_use_responses_api(m.name):
                return _build_responses_api_request(m, messages, output)
            if _is_openai_image_model(m.name):
                return _build_image_generations_request(m, messages, output, self._images_path)
            path, body = _build_openai_compat_request(m, messages, output, self._chat_path, self._mtf)
            if _is_o_series_model(m.name):
                body.pop("temperature", None); body.pop("top_p", None)
            if m.reasoning:
                eff = m._REASONING_MAP.get(m.reasoning)
                if eff: body["reasoning_effort"] = eff
            return path, body

        if p == "xai":
            if _is_xai_image(m.name):
                prompt = _last_user_text(messages)
                return self._images_path, {"model": m.name, "prompt": prompt,
                                           "n": 1, "response_format": "b64_json"}
            path, body = _build_openai_compat_request(m, messages, output, self._chat_path, self._mtf)
            if m.reasoning:
                eff = m._REASONING_MAP.get(m.reasoning)
                if eff: body["reasoning_effort"] = eff
            return path, body

        if p == "perplexity":
            return _build_openai_compat_request(m, messages, output, self._chat_path, self._mtf)

        if p == "kimi":
            path, body = _build_openai_compat_request(m, messages, output, self._chat_path, self._mtf)
            if m.reasoning is not None and m._REASONING_MAP.get(m.reasoning) == "enabled":
                body["temperature"] = 1.0
                body["thinking"] = {"type": "enabled"}
            return path, body

        if p == "deepseek":
            path, body = _build_openai_compat_request(m, messages, output, self._chat_path, self._mtf)
            if m.reasoning is not None:
                body["model"] = m._REASONING_MAP[m.reasoning]
            if _is_deepseek_reasoner(body["model"]):
                body.pop("temperature", None); body.pop("top_p", None)
            fmt = output.get("format", {}); ft = fmt.get("type", "text")
            if ft in ("json", "json_schema"):
                body["response_format"] = {"type": "json_object"}
                has = any("json" in (x.get("content") or "").lower()
                          for x in body["messages"] if x.get("role") == "system")
                if not has:
                    hint = ("Respond with a JSON object that strictly follows this "
                            f"JSON Schema:\n{_json.dumps(fmt['schema'])}"
                            if ft == "json_schema" and fmt.get("schema")
                            else "Respond with a JSON object.")
                    body["messages"] = [{"role": "system", "content": hint}, *body["messages"]]
            return path, body

        if p == "qwen":
            if _is_qwen_image(m.name):
                return _build_image_generations_request(m, messages, output, self._images_path)
            path, body = _build_openai_compat_request(m, messages, output, self._chat_path, self._mtf)
            if _is_qwq(m.name):
                body["enable_thinking"] = True
            elif _is_qwen3(m.name) and m.reasoning:
                body["enable_thinking"] = True
            return path, body

        raise ValueError(f"Unknown openai-family provider {p!r}")

    # ── format: provider response → our result ───────────────────────
    def parse_response(self, response, output) -> "str | dict":
        p = self._provider
        if "data" in response:
            return _parse_image_generations_response(response)
        if p == "openai":
            if "choices" in response:
                return _parse_openai_compat_response(response, output)
            return _parse_responses_api_response(response, output)
        return _parse_openai_compat_response(response, output)


def _last_user_text(messages: list) -> str:
    for msg in reversed(messages):
        if msg["role"] == "user":
            texts = [p["text"] for p in msg["parts"] if p["type"] == "text"]
            if texts:
                return "\n".join(texts)
    return ""
