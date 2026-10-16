"""The Anthropic Messages API family (``POST /v1/messages``), format and
transport: system messages lift to a top-level ``system`` field, structured
output is a forced ``tool_use``, reasoning maps to a thinking budget, and
``cache_control`` places the prompt-cache mark. See docs/models/models.md.
"""

from __future__ import annotations

from ...models._schema import check_structure
import json
import re

from .._base import BaseClient


def _extract_first_json(t: str) -> str:
    """The first balanced top-level JSON object or array in *t*, string literals
    respected: models sometimes wrap the JSON in prose."""
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


def _part_to_anthropic(part: dict) -> dict | None:
    """One universal part as an Anthropic content block, or ``None`` (recorded
    as declined) for a type Claude cannot take."""
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

    from ._openai_compat import _dropped
    if ptype == "document":
        src = part["source"]
        if src.get("kind") in ("base64", "file") and src.get("mime", "application/pdf") == "application/pdf":
            return {"type": "document",
                    "source": {"type": "base64", "media_type": "application/pdf", "data": src["data"]}}
        return _dropped(part, "Claude reads a document as PDF bytes")
    # Claude takes no audio (sent as a `document` it is a 400) and no video:
    # left out, and said.
    if ptype == "image":
        return _dropped(part, f"an image source of kind {part['source'].get('kind')!r} cannot be sent")
    return _dropped(part, f"Claude takes no {ptype} input")




def _note_ttl(params: dict) -> None:
    """The cache lifetime travels inside the conversation, where the check for
    options that never arrived does not look; it is said here instead."""
    from ...models._adaptation import TRANSLATED, Adaptation, record
    if params.get("cache_ttl"):
        record(Adaptation(kind=TRANSLATED, option="cache_ttl", asked=params["cache_ttl"],
                          sent=f"cache_control ttl {params['cache_ttl']}", model=params["name"]))

def _anthropic_id(call_id: str) -> str:
    """A call id in the characters this API accepts. A history from another
    model (Kimi's ``functions.get_weather:0``) is mapped the same way on both
    sides of the turn, so the call and its result still match."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", call_id or "") or "call"

def _encode_messages(messages) -> tuple:
    """Universal messages as Anthropic's system blocks and alternating turns."""
    system_parts: list[dict] = []
    amsgs: list[dict] = []
    for msg in messages:
        role = msg["role"]

        # One call's result: a tool_result block inside a USER message —
        # Anthropic's wire shape. Consecutive results merge into one user
        # message, because the API requires strict role alternation.
        if role == "tool":
            parts = msg.get("parts") or []
            # Anthropic is the one family that takes an image inside the
            # result, so a tool that renders something can be looked at
            # without a second turn.
            if any(p.get("type") != "text" for p in parts):
                blocks = [_part_to_anthropic(p) for p in parts]
                content = [b for b in blocks if b is not None]
            else:
                content = "\n".join(p.get("text", "") for p in parts
                                    if p.get("type") == "text")
            block = {"type": "tool_result",
                     "tool_use_id": _anthropic_id(msg.get("call_id", "")),
                     "content": content}
            if msg.get("is_error"):
                block["is_error"] = True
            if amsgs and amsgs[-1]["role"] == "user" and \
               all(b.get("type") == "tool_result" for b in amsgs[-1]["content"]):
                amsgs[-1]["content"].append(block)
            else:
                amsgs.append({"role": "user", "content": [block]})
            continue

        # The model's own request to act: tool_use blocks, with any
        # accompanying prose as a text block in the same message.
        if role == "assistant" and msg.get("tool_calls"):
            blocks = list((msg.get("opaque") or {}).get("anthropic") or [])
            blocks += [b for b in (_part_to_anthropic(p) for p in msg.get("parts") or [])
                       if b is not None]
            blocks += [{"type": "tool_use", "id": _anthropic_id(c.get("id", "")),
                        "name": c["name"], "input": c.get("arguments", {})}
                       for c in msg["tool_calls"]]
            amsgs.append({"role": "assistant", "content": blocks})
            continue

        blocks = [_part_to_anthropic(p) for p in msg["parts"]]
        blocks = [b for b in blocks if b is not None]
        if not blocks:
            continue
        if role == "system":
            system_parts.extend(blocks)
        else:
            amsgs.append({"role": role, "content": blocks})
    return system_parts, amsgs


def _mark_cache(body: dict, amsgs: list, system_parts: list, params: dict) -> None:
    """Place the prompt-cache breakpoint ``params['cache_control']`` asks for."""
    from ...models._adaptation import Adaptation, ADAPTED, record
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
    # asks the provider to cache content that changes every call.
    #
    # ``cache_control="last"`` is the third placement, and the one an agent
    # loop needs. It marks the last block of the last message, whatever
    # its type — a ``tool_result`` as readily as text. ``True`` looks for a
    # *text* block in the second-to-last message, and in a tool loop that
    # message is the assistant's ``tool_use`` turn, which has none, so
    # nothing would be marked. Marking the newest block is right here,
    # unlike in a chat, because the next call sends exactly this history
    # plus one more turn — the whole of it is the stable prefix.
    mark = params.get("cache_control")
    if mark == "last":
        target = amsgs[-1]["content"] if amsgs else None
        if target:
            control = {"type": "ephemeral"}
            if params.get("cache_ttl") == "1h":
                control["ttl"] = "1h"
            for blk in reversed(target):
                # A thinking block cannot carry a mark; anything else can.
                if isinstance(blk, dict) and blk.get("type") not in (
                        "thinking", "redacted_thinking"):
                    blk["cache_control"] = control
                    _note_ttl(params)
                    record(Adaptation(
                        kind=ADAPTED, option="cache_control", asked=mark,
                        sent="cache_control breakpoint",
                        model=params["name"],
                        why="marked the newest block, so the next call "
                            "reads this whole history back"))
                    break
    by_index = isinstance(mark, int) and not isinstance(mark, bool)
    if mark is True or by_index:
        target = None
        if by_index:
            # Absolute index only, and index 0 is a real answer (a fixed
            # preamble first), so the mark is not tested for truthiness.
            # Python would read -1 as the *last* message, the opposite of
            # "nothing is stable yet". An index that names no message marks
            # nothing: an explicit index is an instruction, not a hint.
            if 0 <= mark < len(amsgs):
                target = amsgs[mark]["content"]
        else:
            if len(amsgs) >= 2:
                target = amsgs[-2]["content"]
            if target is None and system_parts:
                # An all-text system prompt was collapsed into a string
                # above, which has no block to carry the mark. Normalised
                # back, ``cache_control=True`` works for the commonest
                # skill shape — one system prompt, one short user message —
                # where marking that message would cache a changing prefix.
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
                    _note_ttl(params)
                    record(Adaptation(
                        kind=ADAPTED, option="cache_control", asked=mark,
                        sent="cache_control breakpoint",
                        model=params["name"],
                        why="marked the end of the stable prefix; this "
                            "provider caches on an explicit breakpoint"))
                    break


#: What a model takes, read from its registry entry; a model not listed gets
#: the newest generation's mechanisms, from `[provider.mechanisms]`.
_MECHANISMS = ("structured", "reasoning", "thinking_off", "sampling", "forced_tools")


def _mechanisms(data: dict, name: str) -> dict:
    models = data.get("models", {})
    entry = models.get(name)
    if entry is None:
        # An alias reaches its dated entry (claude-sonnet-4-5 is
        # claude-sonnet-4-5-20250929) and a dated release its alias's.
        near = [k for k in models if k.startswith(name + "-") or name.startswith(k + "-")]
        entry = models[max(near, key=len)] if near else {}
    mech = dict(data["provider"].get("mechanisms") or {})
    mech.update({k: v for k, v in entry.items() if k in _MECHANISMS})
    return mech


def _sampling(body: dict, params: dict, output: dict, rmap: dict, default_max: int,
              mech: dict) -> None:
    """Sampling and thinking, by the mechanism the model's data names."""
    from ...models._adaptation import Adaptation, ADAPTED, DECLINED, TRANSLATED, record
    if mech["reasoning"] == "budget":
        if params.get("reasoning") == "none":
            record(Adaptation(kind=TRANSLATED, option="reasoning", asked="none", sent="no thinking",
                              model=params["name"], why="this model does not think unless asked"))
        return _budget_sampling(body, params, output, rmap, default_max, mech)
    name = params["name"]
    if mech["sampling"] == "fixed":
        # Claude 4.7 and later take no sampling controls: a temperature other
        # than 1, or any top_p or top_k, is a 400 ("deprecated for this model").
        temperature = body.pop("temperature", None)
        if temperature not in (None, 1, 1.0):
            record(Adaptation(kind=DECLINED, option="temperature", asked=temperature, sent=None,
                              model=name, why="this model takes no sampling temperature"))
        for option in ("top_p", "top_k"):
            if params.get(option) is not None:
                record(Adaptation(kind=DECLINED, option=option, asked=params[option], sent=None,
                                  model=name, why=f"this model takes no {option}"))
    else:
        if params.get("top_p") is not None:
            body["top_p"] = params["top_p"]
        if params.get("top_k") is not None:
            body["top_k"] = params["top_k"]
    reasoning = params.get("reasoning")
    if reasoning is None:
        return
    if reasoning == "none":
        off = mech["thinking_off"]
        if off == "never":
            body["thinking"] = {"type": "adaptive"}
            body.setdefault("output_config", {})["effort"] = "low"
            record(Adaptation(kind=DECLINED, option="reasoning", asked="none",
                              sent="adaptive thinking at low effort", model=name,
                              why="this model always thinks; the lowest effort was sent"))
        elif off == "between_tools":
            body["thinking"] = {"type": "between_tools"}
            record(Adaptation(kind=ADAPTED, option="reasoning", asked="none", sent="between_tools",
                              model=name, why="this model's lowest setting: no thinking before "
                                              "it answers, short notes between tool calls"))
        else:
            body["thinking"] = {"type": "disabled"}
            record(Adaptation(kind=TRANSLATED, option="reasoning", asked="none",
                              sent="thinking disabled", model=name, why="this provider's own word"))
        return
    body["thinking"] = {"type": "adaptive"}
    body.setdefault("output_config", {})["effort"] = reasoning
    record(Adaptation(kind=ADAPTED, option="reasoning", asked=reasoning,
                      sent=f"adaptive thinking, effort {reasoning}", model=name,
                      why="this model thinks adaptively, steered by an effort level, not a "
                          "token budget"))
    # Adaptive thinking takes temperature 1 or none, and no top_k.
    if body.get("temperature") not in (None, 1, 1.0):
        record(Adaptation(kind=DECLINED, option="temperature", asked=body.pop("temperature"),
                          sent=None, model=name, why="thinking takes no other temperature"))
    if "top_k" in body:
        record(Adaptation(kind=DECLINED, option="top_k", asked=body.pop("top_k"), sent=None,
                          model=name, why="thinking takes no top_k"))


def _budget_sampling(body: dict, params: dict, output: dict, rmap: dict, default_max: int,
                     mech: dict) -> None:
    """Temperature, top_p, top_k and a thinking budget, as Claude 4.6 and earlier take them."""
    from ...models._adaptation import Adaptation, ADAPTED, DECLINED, record
    reasoning = params.get("reasoning")
    if reasoning == "none":
        reasoning = None                 # thinking is off here unless asked for
    # The API's own range for temperature is 0 to 1; above it is a 400
    # after the round trip, where it can be said here.
    if (body.get("temperature") or 0) > 1.0:
        record(Adaptation(
            kind=ADAPTED, option="temperature", asked=body["temperature"], sent=1.0,
            model=params["name"], why="this provider's range is 0 to 1"))
        body["temperature"] = 1.0
    if params.get("top_p") is not None:
        body["top_p"] = params["top_p"]
        # Claude takes one of the two, not both: the request is refused
        # when both are set. The one the caller set by name wins.
        if "temperature" in body:
            record(Adaptation(
                kind=ADAPTED, option="temperature", asked=body.pop("temperature"),
                sent=None, model=params["name"],
                why="this provider takes temperature or top_p, not both; top_p was set"))
    if params.get("top_k") is not None:
        body["top_k"] = params["top_k"]

    if reasoning:
        record(Adaptation(
            kind=ADAPTED, option="reasoning", asked=reasoning,
            sent="thinking", model=params["name"],
            why="sent as a thinking budget, this provider's own control"))
        budget = rmap.get(reasoning)
        if budget is not None:
            if output.get("format", {}).get("type") == "json_schema" \
                    and mech["structured"] == "forced_tool":
                # Structured output here is a forced tool call, and the API
                # does not take a forced tool with thinking on: refused
                # before the round trip, as tools with a schema are below.
                raise ValueError(
                    f"{params['name']} cannot combine reasoning with output json_schema: on this "
                    "model structured output is a forced tool call, which extended thinking does "
                    "not allow. Drop reasoning, or ask for format 'json'.")
            body["thinking"]    = {"type": "enabled", "budget_tokens": budget}
            if body.get("temperature") not in (None, 1.0):
                record(Adaptation(
                    kind=DECLINED, option="temperature",
                    asked=body["temperature"], sent=1.0,
                    model=params["name"],
                    why="this provider requires temperature 1.0 while "
                        "thinking is enabled"))
            body["temperature"] = 1.0
            if "top_p" in body:
                # Thinking takes top_p only from 0.95, and not beside a
                # temperature: the temperature it requires is the default.
                body.pop("temperature")
                if body["top_p"] < 0.95:
                    record(Adaptation(
                        kind=ADAPTED, option="top_p", asked=body["top_p"], sent=0.95,
                        model=params["name"],
                        why="extended thinking takes top_p from 0.95 to 1"))
                    body["top_p"] = 0.95
            if "top_k" in body:
                record(Adaptation(
                    kind=DECLINED, option="top_k", asked=body.pop("top_k"), sent=None,
                    model=params["name"],
                    why="extended thinking does not take top_k"))
            if body["max_tokens"] <= budget:
                # The budget has to fit inside the answer's ceiling, so a
                # max_tokens smaller than it is raised rather than left to
                # fail at the provider.
                record(Adaptation(
                    kind=ADAPTED, option="max_tokens",
                    asked=body["max_tokens"], sent=budget + default_max,
                    model=params["name"],
                    why="raised above the thinking budget, which the "
                        "answer ceiling has to contain"))
                body["max_tokens"] = budget + default_max


#: JSON Schema keywords structured output does not take; dropped before the
#: call, still checked on the answer against the caller's schema.
_UNSUPPORTED = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
                "minLength", "maxLength", "maxItems", "uniqueItems", "minProperties",
                "maxProperties", "patternProperties")
_FORMATS = {"date-time", "time", "date", "duration", "email", "hostname", "uri", "ipv4",
            "ipv6", "uuid"}
MAX_OPTIONAL, MAX_UNIONS = 24, 16


def _for_anthropic(schema: dict, name: str) -> dict:
    """*schema* as structured output takes it: unsupported keywords removed,
    every object closed, and the documented limits refused before the call."""
    from ...models._adaptation import Adaptation, ADAPTED, record
    removed: list = []
    counts = {"optional": 0, "unions": 0}

    def walk(node, path):
        if isinstance(node, list):
            return [walk(x, f"{path}[{i}]") for i, x in enumerate(node)]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str) and not ref.startswith("#"):
            raise ValueError(f"{name}: structured output takes no external $ref ({ref} at {path})")
        out = {}
        for key, value in node.items():
            if key in _UNSUPPORTED or (key == "minItems" and value not in (0, 1)) \
                    or (key == "format" and value not in _FORMATS):
                removed.append(f"{path}.{key}")
                continue
            if key == "additionalProperties" and value is not False:
                removed.append(f"{path}.additionalProperties")
                continue
            if key in ("properties", "$defs", "definitions"):
                out[key] = {k: walk(v, f"{path}.{k}") for k, v in value.items()}
            else:
                out[key] = walk(value, f"{path}.{key}")
        if node.get("type") == "object" or "properties" in node:
            out["additionalProperties"] = False
            required = set(node.get("required") or [])
            counts["optional"] += sum(1 for k in (node.get("properties") or {}) if k not in required)
        if "anyOf" in node or isinstance(node.get("type"), list):
            counts["unions"] += 1
        return out

    # Refs resolve against the whole schema, so a cycle is found by following them.
    defs = {**(schema.get("$defs") or {}), **(schema.get("definitions") or {})}

    def cycle(node, stack):
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/"):
                key = ref.rsplit("/", 1)[-1]
                if key in stack:
                    raise ValueError(f"{name}: structured output takes no recursive schema "
                                     f"({ref} refers back to itself)")
                if key in defs:
                    cycle(defs[key], stack | {key})
            for v in node.values():
                cycle(v, stack)
        elif isinstance(node, list):
            for v in node:
                cycle(v, stack)

    cycle(schema, frozenset())
    sent = walk(schema, "$")
    if counts["optional"] > MAX_OPTIONAL:
        raise ValueError(f"{name}: structured output takes at most {MAX_OPTIONAL} optional "
                         f"properties; this schema has {counts['optional']} — make some required")
    if counts["unions"] > MAX_UNIONS:
        raise ValueError(f"{name}: structured output takes at most {MAX_UNIONS} union types "
                         f"(anyOf or a list of types); this schema has {counts['unions']}")
    if removed:
        record(Adaptation(kind=ADAPTED, option="format.schema", asked=", ".join(removed),
                          sent="removed", model=name,
                          why="structured output does not take these; the answer is still "
                              "checked against them"))
    return sent


def _answer_format(body: dict, fmt: dict, tools, name: str, mech: dict) -> None:
    """Tools and the answer's format: JSON by instruction, a schema by the
    mechanism the model takes — `output_config.format`, or a forced tool."""
    from ...models._adaptation import Adaptation, ADAPTED, record
    forced = fmt.get("type") == "json_schema" and mech["structured"] == "forced_tool"
    if tools:
        if forced:
            raise ValueError(
                f"{name} cannot combine native tools with output json_schema: on this model "
                "structured output is a forced tool call, which would shadow the real tools. "
                "Drop one of the two.")
        body["tools"] = [
            {"name": t["function"]["name"],
             "description": t["function"].get("description", ""),
             "input_schema": t["function"].get("parameters",
                                               {"type": "object"})}
            if "function" in t else t
            for t in tools
        ]
    if fmt.get("type") == "json":
        # Anthropic has no JSON-mode field. Every other provider does, so
        # this is absorbed rather than refused: the intent is portable,
        # only the control is missing here.
        #
        # An instruction, not a prefilled `{`: prefilling is Anthropic's
        # own trick and it works, but the opening brace is then missing
        # from the reply, so every reader of the raw text — ours and the
        # caller's — has to know it was done.
        rule = ("Respond with a single valid JSON object and nothing "
                "else: no prose, no code fence.")
        existing = body.get("system")
        if isinstance(existing, str):
            body["system"] = f"{existing}\n\n{rule}"
        elif isinstance(existing, list):
            body["system"] = existing + [{"type": "text", "text": rule}]
        else:
            body["system"] = rule
        record(Adaptation(
            kind=ADAPTED, option="format.json", asked="json",
            sent="a system instruction", model=name,
            why="anthropic has no JSON-mode field; the requirement is "
                "carried in the system prompt instead"))


    if fmt.get("type") == "json_schema" and not forced:
        body.setdefault("output_config", {})["format"] = {
            "type": "json_schema", "schema": _for_anthropic(fmt["schema"], name)}
    elif forced:
        tool = fmt.get("name", "structured_output")
        spec = {"name": tool,
                "description": "Return the result matching the given schema.",
                "input_schema": fmt["schema"]}
        # Without this the schema is advisory: the provider forces the tool
        # call but not its shape, so a model may fill one field and drop
        # the rest.
        if fmt.get("strict"):
            spec["strict"] = True
        body["tools"] = [spec]
        if mech.get("forced_tools"):
            body["tool_choice"] = {"type": "tool", "name": tool}
        else:
            # A model that refuses a forced choice: the tool is offered, strict,
            # and the requirement is said, since nothing else can force it.
            spec["strict"] = True
            body["tool_choice"] = {"type": "auto"}
            rule = f"Answer only by calling the {tool} tool."
            existing = body.get("system")
            body["system"] = (f"{existing}\n\n{rule}" if isinstance(existing, str)
                              else existing + [{"type": "text", "text": rule}] if isinstance(existing, list)
                              else rule)
            record(Adaptation(kind=ADAPTED, option="format.json_schema", asked="a forced tool",
                              sent="tool_choice auto, a strict tool and an instruction", model=name,
                              why="this model takes no forced tool choice"))


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
        return {**super()._auth_headers(), "anthropic-version": _API_VERSION}

    def list_models(self) -> list[str]:
        # Every page: the first holds only twenty models.
        out, after = [], None
        for _ in range(50):
            path = "/v1/models?limit=1000" + (f"&after_id={after}" if after else "")
            page = json.loads(self._get(path, self._auth_headers()))
            out += [m["id"] for m in page.get("data") or []]
            if not page.get("has_more") or not page.get("last_id"):
                break
            after = page["last_id"]
        return out

    # ── format ───────────────────────────────────────────────────────
    #: Native tool calling implemented: tool_use / tool_result blocks.
    supports_tools = True

    supports_streaming = True

    def build_stream_request(self, messages, output, params, tools=None):
        path, body = self.build_request(messages, output, params, tools=tools)
        body["stream"] = True
        return path, body

    def parse_stream_event(self, event: dict, output: dict) -> str | None:
        # Anthropic names its event types rather than sending one shape with
        # empty fields, so the filter is exact: only a text delta is text.
        # A thinking delta arrives under the same event type with a different
        # delta type, and passing it through would splice the model's private
        # reasoning into the answer the caller shows a user.
        if event.get("type") != "content_block_delta":
            return None
        delta = event.get("delta") or {}
        if delta.get("type") != "text_delta":
            return None
        text = delta.get("text")
        return text if isinstance(text, str) and text else None

    def stream_tool_fragments(self, event: dict) -> list | None:
        """Anthropic opens a content block for the call and then streams its
        arguments into it, so the name arrives in one event type and the
        arguments in another — under `partial_json`, not `text`, which is why
        `parse_stream_event` filtering on `text_delta` does not see them."""
        index = event.get("index")
        etype = event.get("type")
        if etype == "content_block_start":
            block = event.get("content_block") or {}
            if block.get("type") != "tool_use":
                return []
            return [{"slot": index, "id": block.get("id") or "",
                     "name": block.get("name") or "", "arguments": ""}]
        if etype == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") != "input_json_delta":
                return []
            return [{"slot": index,
                     "arguments": delta.get("partial_json") or ""}]
        return []

    def stream_usage(self, event: dict) -> dict | None:
        """
        Anthropic reports usage **twice**, and the halves are different.

        `message_start` carries the input tokens; `message_delta` carries the
        output tokens and nothing else. Taking the last report — the obvious
        implementation, and the one every other family here needs — loses the
        whole prompt and under-bills by exactly the input side, invisibly,
        because the number that comes back is still a plausible number.

        So each half is returned as it arrives and the merge happens in the
        caller, which accumulates rather than replaces for this reason.
        """
        etype = event.get("type")
        if etype == "message_start":
            usage = ((event.get("message") or {}).get("usage")) or {}
            return {"usage": dict(usage)} if usage else None
        if etype == "message_delta":
            usage = event.get("usage") or {}
            return {"usage": dict(usage)} if usage else None
        return None

    def build_request(self, messages, output, params, tools=None) -> tuple[str, dict]:
        system_parts, amsgs = _encode_messages(messages)
        body: dict = {
            "model":       params["name"],
            "messages":    amsgs,
            "max_tokens":  params["max_tokens"],
            "temperature": params["temperature"],
        }
        if system_parts:
            if all(b["type"] == "text" for b in system_parts):
                body["system"] = "\n\n".join(b["text"] for b in system_parts)
            else:
                body["system"] = system_parts
        _mark_cache(body, amsgs, system_parts, params)
        prov = self._data["provider"]
        mech = _mechanisms(self._data, params["name"])
        _sampling(body, params, output, prov.get("reasoning_map", {}), prov["defaults"]["max_tokens"],
                  mech)
        _answer_format(body, output.get("format", {}), tools, params["name"], mech)
        return "/v1/messages", body

    def parse_response(self, response, output) -> str | dict:
        ftype   = output.get("format", {}).get("type", "text")
        content = response.get("content", [])
        answer_tool = output.get("format", {}).get("name", "structured_output")
        if ftype == "json_schema" and not any(
                b.get("type") == "tool_use" and b.get("name") != answer_tool for b in content):
            # Answered in the text (`output_config.format`) or through the
            # answer tool; a call to one of the caller's tools is a call, and
            # goes on below. No answer — cut off, or refused — raises, as on
            # the other families, rather than returning `{}`.
            schema = (output.get("format") or {}).get("schema")
            block = next((b for b in content if b.get("type") == "tool_use"), None)
            text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
            if block is None:
                from .._errors import InvalidStructuredOutputError, TruncatedResponseError
                spent = (response.get("usage") or {}).get("output_tokens")
                if response.get("stop_reason") == "refusal":
                    raise InvalidStructuredOutputError(
                        f"the model refused to answer (stop_reason='refusal'): {text[:300]}",
                        problems=["refusal"], value=None)
                if response.get("stop_reason") == "max_tokens":
                    raise TruncatedResponseError(
                        f"Response was cut off at {spent} output tokens before the structured "
                        f"answer was written — raise max_tokens above {spent}.",
                        finish_reason="max_tokens", output_tokens=spent)
                if not text.strip():
                    raise InvalidStructuredOutputError(
                        "the model gave no structured answer (stop_reason="
                        f"{response.get('stop_reason')!r})", value=None)
                # `output_config.format`: the answer is the text, checked
                # against the caller's schema, not the one the provider was sent.
                from ._structured import parse_structured
                return parse_structured(text, schema, strip_fences=True)
            value = block.get("input") or {}
            if isinstance(schema, dict) and (problems := check_structure(value, schema)):
                from .._errors import InvalidStructuredOutputError
                raise InvalidStructuredOutputError(
                    "the structured answer does not match the schema: " + "; ".join(problems),
                    problems=problems, value=value)
            return value
        # Every text block, joined: Claude splits an answer around citations
        # and server tools, and the first block alone is a fragment.
        text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
        calls = [b for b in content if b.get("type") == "tool_use"]
        from ._structured import refuse_cut_off
        refuse_cut_off(response.get("stop_reason") or "",
                       (response.get("usage") or {}).get("output_tokens"),
                       calls=bool(calls), text=text)
        if calls:
            from ...models._calls import ToolCall, ToolCallRequest
            # With extended thinking on, the turn that asked for the calls has
            # to come back opening with its thinking blocks, unchanged — the
            # signature is checked — or the next request is a 400.
            thinking = [b for b in content if b.get("type") in ("thinking", "redacted_thinking")]
            return ToolCallRequest(
                calls=tuple(ToolCall(id=b.get("id", ""), name=b.get("name", ""),
                                     arguments=b.get("input") or {})
                            for b in calls),
                text=text,
                opaque={"anthropic": thinking} if thinking else {},
            )
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
