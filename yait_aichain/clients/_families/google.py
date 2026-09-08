"""
clients._families.google
=========================

Google Generative AI family (``POST /models/{model}:generateContent``):
format + transport (x-goog-api-key).

System messages lift into ``system_instruction``; generation params live in
``generationConfig``; reasoning maps to ``thinkingConfig``; image output uses
``responseModalities``.
"""

from __future__ import annotations

import json

from .._base import BaseClient


#: One warning per process is enough to be seen and few enough to be ignored
#: inside a loop; the run would otherwise print it once per call.
_WARNED_STRIP = False


def _sanitize_google_schema(schema: object) -> object:
    """
    Convert a JSON Schema dict to a form accepted by Google's ``responseSchema``
    proto field.

    Three incompatibilities are fixed recursively:

    * ``additionalProperties``          — not supported; stripped, with one
      warning per process. It is not cosmetic: ``additionalProperties: false``
      is how a schema says "no fields beyond these", so removing it inverts
      the instruction. The model then returns extra fields and nothing
      notices — the failure is only visible by diffing two providers' output
      against the same schema. ``check_structure`` catches it on the way
      back; the warning is so it is caught before the run.
    * ``"type": ["X", "null"]``         — union types are not supported;
      converted to ``"type": "X", "nullable": true``.
    * ``"type": "array"`` with no ``items`` — Google requires the element
      schema and rejects the whole request without it, before the first step.
      A permissive default is filled in rather than failing: every other
      provider accepts the loose form, so a tool that works elsewhere must
      not be the reason a run cannot start here.
    """
    if not isinstance(schema, dict):
        return schema

    result: dict = {}
    for key, value in schema.items():
        if key == "additionalProperties":
            global _WARNED_STRIP
            if value is False and not _WARNED_STRIP:
                _WARNED_STRIP = True
                import warnings
                warnings.warn(
                    "Google does not accept 'additionalProperties', so it was "
                    "removed from the schema — the model may return fields "
                    "the schema does not declare. Structured replies are "
                    "checked on the way back and will raise "
                    "InvalidStructuredOutputError if it does. Use "
                    "yait_aichain.portable_schema(schema, 'google') to see "
                    "the form actually sent.",
                    RuntimeWarning, stacklevel=3,
                )
            continue                              # strip
        if key == "type" and isinstance(value, list):
            non_null = [t for t in value if t != "null"]
            result["type"] = non_null[0] if non_null else "string"
            if "null" in value:
                result["nullable"] = True
        elif key == "properties" and isinstance(value, dict):
            result[key] = {k: _sanitize_google_schema(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, dict):
            result[key] = _sanitize_google_schema(value)
        else:
            result[key] = value
    if result.get("type") == "array" and "items" not in result:
        result["items"] = {"type": "string"}
    return result


def _part_to_google(part: dict) -> "dict | None":
    """
    Convert one universal part dict to a Google AI part object.

    Returns ``None`` for unsupported source kinds so callers can filter.
    """
    ptype = part["type"]

    if ptype == "text":
        return {"text": part["text"]}

    if ptype == "image":
        src  = part["source"]
        kind = src["kind"]
        if kind == "url":
            return {
                "fileData": {
                    "mimeType": src.get("mime", "image/png"),
                    "fileUri":  src["url"],
                },
            }
        if kind in ("base64", "file"):
            return {
                "inlineData": {
                    "mimeType": src.get("mime", "image/png"),
                    "data":     src["data"],
                },
            }

    if ptype == "video":
        src  = part["source"]
        kind = src["kind"]
        if kind == "url":
            return {
                "fileData": {
                    "mimeType": src.get("mime", "video/mp4"),
                    "fileUri":  src["url"],
                },
            }
        if kind in ("base64", "file"):
            return {
                "inlineData": {
                    "mimeType": src.get("mime", "video/mp4"),
                    "data":     src["data"],
                },
            }

    if ptype == "audio":
        src  = part["source"]
        kind = src["kind"]
        if kind == "base64":
            return {
                "inlineData": {
                    "mimeType": src.get("mime", "audio/wav"),
                    "data":     src["data"],
                },
            }
        if kind == "url":
            return {
                "fileData": {
                    "mimeType": src.get("mime", "audio/wav"),
                    "fileUri":  src["url"],
                },
            }

    return None


class GoogleClient(BaseClient):

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
        return {"Content-Type": "application/json",
                "x-goog-api-key": self._api_key}

    def list_models(self) -> list[str]:
        data = self._get("/models", self._auth_headers())
        return [m["name"].removeprefix("models/") for m in json.loads(data)["models"]]

    # ── format ───────────────────────────────────────────────────────
    #: Native tool calling implemented: functionCall / functionResponse
    #: parts. Gemini keys results by NAME, not id — there are no call ids
    #: on this wire, so ToolCall.id degrades to the function name.
    supports_tools = True

    def build_request(self, messages, output, params, tools=None) -> "tuple[str, dict]":
        prov = self._data["provider"]
        rmap = prov.get("reasoning_map", {})
        name = params["name"]

        system_parts: list[dict] = []
        contents: list[dict] = []
        for msg in messages:
            # One call's result: a functionResponse part in a USER turn.
            if msg["role"] == "tool":
                # A functionResponse carries JSON, not media. Anything a tool
                # rendered is moved into a user message right after it, so the
                # model still sees it on this turn instead of reading a
                # description of it.
                from ...models._calls import split_media_result
                msg, follow_up = split_media_result(msg)
                text = "\n".join(p.get("text", "")
                                  for p in msg.get("parts") or []
                                  if p.get("type") == "text")
                contents.append({"role": "user", "parts": [{
                    "functionResponse": {
                        # call_id here carries the function NAME (see class
                        # comment): Gemini has no ids on this wire.
                        "name": msg.get("call_id", ""),
                        "response": {"result": text},
                    }}]})
                if follow_up:
                    gparts = [_part_to_google(p) for p in follow_up["parts"]]
                    gparts = [g for g in gparts if g is not None]
                    if gparts:
                        contents.append({"role": "user", "parts": gparts})
                continue
            # The model's request to act: functionCall parts on a model turn.
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                gparts = [_part_to_google(p) for p in msg.get("parts") or []]
                gparts = [g for g in gparts if g is not None]
                gparts += [{"functionCall": {"name": c["name"],
                                             "args": c.get("arguments", {})}}
                           for c in msg["tool_calls"]]
                contents.append({"role": "model", "parts": gparts})
                continue
            gparts = [_part_to_google(p) for p in msg["parts"]]
            gparts = [g for g in gparts if g is not None]
            if not gparts:
                continue
            if msg["role"] == "system":
                system_parts.extend(gparts)
            else:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": gparts})

        gc: dict = {"temperature": params["temperature"],
                    "maxOutputTokens": params["max_tokens"]}
        if params.get("top_p") is not None:
            gc["topP"] = params["top_p"]
        if params.get("top_k") is not None:
            gc["topK"] = params["top_k"]
        if params.get("reasoning"):
            budget = rmap.get(params["reasoning"])
            if budget is not None:
                gc["thinkingConfig"] = {"thinkingBudget": budget}

        fmt = output.get("format", {})
        ftype = fmt.get("type", "text")
        modalities = output.get("modalities", ["text"])
        if "image" in modalities:
            mods = ["IMAGE"] + (["TEXT"] if "text" in modalities else [])
            gc["responseModalities"] = mods
        elif ftype == "json":
            gc["responseMimeType"] = "application/json"
        elif ftype == "json_schema":
            gc["responseMimeType"] = "application/json"
            gc["responseSchema"] = _sanitize_google_schema(fmt["schema"])

        body: dict = {"contents": contents, "generationConfig": gc}
        if tools:
            body["tools"] = [{"functionDeclarations": [
                {"name": t["function"]["name"],
                 "description": t["function"].get("description", ""),
                 "parameters": _sanitize_google_schema(
                     t["function"].get("parameters", {"type": "object"}))}
                if "function" in t else t
                for t in tools
            ]}]
        if system_parts:
            body["system_instruction"] = {"parts": system_parts}
        return f"/models/{name}:generateContent", body

    def parse_response(self, response, output) -> "str | dict":
        ftype = output.get("format", {}).get("type", "text")
        cands = response.get("candidates", [])
        if not cands:
            if ftype == "image":
                block = (response.get("promptFeedback") or {}).get("blockReason")
                raise ValueError(
                    "Google returned no image — the request was blocked or refused"
                    + (f": {block}" if block else ".")
                )
            return ""
        finish = cands[0].get("finishReason")
        parts  = cands[0].get("content", {}).get("parts", [])
        if ftype == "image":
            ip = next((p for p in parts if "inlineData" in p), None)
            if ip:
                return {"url": None, "base64": ip["inlineData"]["data"],
                        "mime_type": ip["inlineData"].get("mimeType", "image/png"),
                        "revised_prompt": ""}
            raise ValueError(
                "Google returned no image part — the model emitted text instead "
                "of an image, or the request was blocked"
                + (f" (finish_reason={finish})" if finish else ".")
            )
        calls = [p["functionCall"] for p in parts if "functionCall" in p]
        try:
            text = next(p["text"] for p in parts if "text" in p)
        except StopIteration:
            text = ""
        if calls:
            from ...models._calls import ToolCall, ToolCallRequest
            return ToolCallRequest(
                calls=tuple(ToolCall(id=c.get("name", ""),
                                     name=c.get("name", ""),
                                     arguments=c.get("args") or {})
                            for c in calls),
                text=text,
            )
        if ftype in ("json", "json_schema"):
            # A response cut off at the token ceiling arrives as HTTP 200 with
            # valid-looking JSON that simply stops; a schema this client had to
            # strip `additionalProperties` from comes back whole and wrong.
            # Both are HTTP 200 and the fixes are opposite, so they are
            # separated here rather than left to the caller's JSONDecodeError.
            from ._structured import parse_structured
            meta = response.get("usageMetadata") or {}
            return parse_structured(
                text, output.get("format", {}).get("schema"),
                finish_reason=finish or "",
                output_tokens=meta.get("candidatesTokenCount"),
                strip_fences=True,
            )
        return text
