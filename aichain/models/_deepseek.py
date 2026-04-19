"""
models._deepseek — DeepSeekModel
==================================

Covers the DeepSeek model family:

  Chat / text  (Chat Completions — /v1/chat/completions)
  ────────────
  deepseek-chat       Standard chat model (DeepSeek-V3).
                      128K context, full parameter support.

  deepseek-reasoner   Always-on Chain-of-Thought model (DeepSeek-R1).
                      128K context.  Returns reasoning traces separately
                      in ``reasoning_content``; final answer in ``content``.

Model behaviour differences
───────────────────────────

  Parameter              deepseek-chat     deepseek-reasoner
  ─────────────────────────────────────────────────────────────
  temperature            ✓ 0–2             ✗ ignored by API
  top_p                  ✓                 ✗ ignored by API
  presence_penalty       ✓                 ✗ ignored by API
  frequency_penalty      ✓                 ✗ ignored by API
  max_tokens             ✓ (default 4096)  ✓ (default 32768, max 64K,
                                             covers CoT + answer)
  tools / function call  ✓                 ✗ not supported
  response_format        ✓ json_object /   ✓ json_object /
                           json_schema       json_schema

For ``deepseek-reasoner``, ``temperature``, ``top_p``,
``presence_penalty``, and ``frequency_penalty`` are silently ignored by
the DeepSeek API (no error is raised, but the values have no effect).
The library omits them from the request body entirely when the model is
``deepseek-reasoner`` to keep requests clean.

Reasoning / CoT output
───────────────────────
``deepseek-reasoner`` always produces a Chain-of-Thought trace.  The
trace is returned in a separate ``reasoning_content`` field alongside
the regular ``content`` field in the response.

The library extracts only the final ``content`` by default, consistent
with every other provider.  If you need the raw reasoning trace, access
it via ``skill.run(...)``'s raw response or read it directly from the
model's HTTP response.

Universal ``reasoning`` option mapping
───────────────────────────────────────
  Universal level   Effect
  ─────────────────────────────────────────────────────────────────
  None              Routing is unchanged.  ``deepseek-chat`` runs
                    normally; ``deepseek-reasoner`` always uses CoT.
  "low"             Selects ``deepseek-chat`` (no CoT).
  "medium"          Selects ``deepseek-chat`` (no CoT).
  "high"            Selects ``deepseek-reasoner`` (always-on CoT).

Because DeepSeek's two models *are* the reasoning toggle, the mapping
is expressed as a model-name override rather than an API parameter.
When ``reasoning="high"`` is set on a ``deepseek-chat`` instance, the
library routes the request to ``deepseek-reasoner`` automatically, and
vice versa.

Default generation parameters
──────────────────────────────
temperature   0.0    DeepSeek recommends 0 for coding/math, 1 for
                     general chat.  0 is a safe conservative default.
max_tokens    4096   Practical default for deepseek-chat.
                     Override to 32768 for deepseek-reasoner to
                     accommodate long CoT traces.
top_p         1.0    DeepSeek default.
top_k         None   Not in the DeepSeek API parameter set.
cache_control False  DeepSeek uses automatic disk-cache for repeated
                     prefixes; there is no explicit cache_control param.
reasoning     None   See mapping above.

Environment variable
────────────────────
DEEPSEEK_API_KEY
"""

import json

import urllib3

from ._base import Model
from ._openai import _part_to_openai, _parse_openai_compat_response
from clients._deepseek import DeepSeekClient
from clients._constants import DEFAULT_TIMEOUT, DEFAULT_RETRIES


# Unsupported parameters for deepseek-reasoner — omit from request body
# to keep payloads clean (API silently ignores them anyway).
_REASONER_UNSUPPORTED = frozenset({"temperature", "top_p",
                                   "presence_penalty", "frequency_penalty"})

# Model name that activates always-on CoT
_REASONER_MODEL = "deepseek-reasoner"
_CHAT_MODEL     = "deepseek-chat"


def _is_reasoner(name: str) -> bool:
    """Return True when *name* is the always-on CoT model."""
    return name == _REASONER_MODEL


# ---------------------------------------------------------------------------
# DeepSeekModel
# ---------------------------------------------------------------------------

class DeepSeekModel(Model):
    """Provider-specific model for the DeepSeek API."""

    _ENV_KEY = "DEEPSEEK_API_KEY"

    # ── generation defaults ──────────────────────────────────────────────
    _DEFAULT_TEMPERATURE:   float        = 0.0
    _DEFAULT_MAX_TOKENS:    int          = 4096
    _DEFAULT_TOP_P:         float | None = 1.0
    _DEFAULT_TOP_K:         int   | None = None   # unsupported
    _DEFAULT_CACHE_CONTROL: bool         = False
    _DEFAULT_REASONING:     str   | None = None

    # reasoning level → effective model name
    # "low" / "medium" stay on deepseek-chat; "high" switches to deepseek-reasoner.
    # The actual API parameter is handled in to_request() via model-name routing.
    _REASONING_MAP: dict = {
        "low":    _CHAT_MODEL,
        "medium": _CHAT_MODEL,
        "high":   _REASONER_MODEL,
    }

    # ------------------------------------------------------------------

    def _build_client(self, api_key: str, client_options: dict) -> DeepSeekClient:
        """
        Construct a :class:`~clients.DeepSeekClient`.

        Supported *client_options* keys: ``url``, ``timeout``,
        ``retries``, ``proxy``.
        """
        return DeepSeekClient(
            api_key = api_key,
            url     = client_options.get("url"),
            timeout = client_options.get("timeout", DEFAULT_TIMEOUT),
            retries = client_options.get("retries", DEFAULT_RETRIES),
            proxy   = client_options.get("proxy"),
        )

    def to_request(self, messages: list, output: dict) -> "tuple[str, dict]":
        """
        Translate universal messages into a DeepSeek ``/v1/chat/completions``
        request.

        Model routing
        -------------
        When ``self.reasoning`` is ``"high"``, the effective model name is
        overridden to ``deepseek-reasoner`` regardless of the name the
        instance was constructed with.  When ``"low"`` or ``"medium"``, the
        effective model is ``deepseek-chat``.  When ``None``, the model name
        is used as-is.

        deepseek-reasoner constraints
        -----------------------------
        ``temperature``, ``top_p``, ``presence_penalty``, and
        ``frequency_penalty`` are omitted from the request body for
        ``deepseek-reasoner`` — the API ignores them silently, but omitting
        them keeps payloads clean and avoids future deprecation warnings.
        ``max_tokens`` is included for both models; the caller should raise
        it to 32 768+ when using the reasoner to leave room for the CoT.
        """
        # ── Resolve effective model name ──────────────────────────────
        if self.reasoning is not None:
            effective_model = self._REASONING_MAP[self.reasoning]
        else:
            effective_model = self.name

        reasoner = _is_reasoner(effective_model)

        # ── Convert universal messages ────────────────────────────────
        openai_messages: list[dict] = []
        for msg in messages:
            role  = msg["role"]
            items = [_part_to_openai(p) for p in msg["parts"]]
            items = [it for it in items if it is not None]
            if not items:
                continue
            if len(items) == 1 and items[0]["type"] == "text":
                openai_messages.append({"role": role, "content": items[0]["text"]})
            else:
                openai_messages.append({"role": role, "content": items})

        # ── Build request body ────────────────────────────────────────
        body: dict = {
            "model":      effective_model,
            "messages":   openai_messages,
            "max_tokens": self.max_tokens,
        }

        # Temperature and sampling params are ignored by the reasoner;
        # omit them for clean requests.
        if not reasoner:
            body["temperature"] = self.temperature
            if self.top_p is not None:
                body["top_p"] = self.top_p

        # ── Output format ─────────────────────────────────────────────
        fmt   = output.get("format", {})
        ftype = fmt.get("type", "text")
        if ftype in ("json", "json_schema"):
            # DeepSeek requires the word "json" to appear somewhere in the
            # conversation when response_format=json_object is set.  For
            # json_schema we fall back to json_object mode because DeepSeek's
            # API does not support the json_schema response_format type.
            body["response_format"] = {"type": "json_object"}
            has_json_word = any(
                "json" in (m.get("content") or "").lower()
                for m in body["messages"]
                if m.get("role") == "system"
            )
            if not has_json_word:
                # Build a compact schema hint so the model knows exactly what
                # JSON structure to produce.  For json_schema we serialise the
                # schema itself; for plain json a minimal instruction suffices.
                if ftype == "json_schema" and fmt.get("schema"):
                    import json as _json
                    hint = (
                        "Respond with a JSON object that strictly follows this "
                        f"JSON Schema:\n{_json.dumps(fmt['schema'])}"
                    )
                else:
                    hint = "Respond with a JSON object."
                body["messages"] = [
                    {"role": "system", "content": hint},
                    *body["messages"],
                ]

        return "/v1/chat/completions", body

    def from_response(self, response: dict, output: dict) -> "str | dict":
        """
        Extract the clean result from a DeepSeek Chat Completions response.

        Both ``deepseek-chat`` and ``deepseek-reasoner`` return the final
        answer in ``choices[0].message.content``.  The CoT trace is in the
        separate ``choices[0].message.reasoning_content`` field, which the
        library discards by default (consistent with all other providers).

        Use the standard OpenAI-compatible extractor.
        """
        return _parse_openai_compat_response(response, output)
