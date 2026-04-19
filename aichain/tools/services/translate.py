"""
tools.services.translate — serviceTranslate
=============================================

Merges DeepLTranslateTool and DeepLRephraseTool into a single class with two
methods: ``run()`` for translation and ``rephrase()`` for style improvement.

Both functions authenticate with a single ``DEEPL_API_KEY`` environment
variable.  Free-tier keys (ending with ``:fx``) are routed to
``api-free.deepl.com`` automatically; Pro keys go to ``api.deepl.com``.

Installation
------------
    pip install urllib3   # already required by other tools

Docs
----
  Translate : https://developers.deepl.com/api-reference/translate
  Rephrase  : https://developers.deepl.com/api-reference/improve-text

Environment variable
--------------------
``DEEPL_API_KEY`` — API key from https://www.deepl.com/pro-api
"""

from __future__ import annotations

import json
import os

import urllib3

from .._base import Tool


# ── Internal helpers ───────────────────────────────────────────────────────────

_FREE_BASE = "https://api-free.deepl.com/v2"
_PRO_BASE  = "https://api.deepl.com/v2"
_ENV_KEY   = "DEEPL_API_KEY"


def _api_base(key: str) -> str:
    """Route free-tier keys (ending with ':fx') to the free endpoint."""
    return _FREE_BASE if key.endswith(":fx") else _PRO_BASE


# ── serviceTranslate ───────────────────────────────────────────────────────────

class serviceTranslate(Tool):
    """
    Translate or rephrase text using the DeepL API.

    Use :meth:`run` for translation between languages and :meth:`rephrase`
    to improve style and clarity within the same language.

    The source language is detected automatically when *source_lang* is
    omitted.  Free-tier and Pro keys are both supported — the correct
    endpoint is chosen from the key suffix automatically.

    Parameters
    ----------
    api_key : str | None, optional
        DeepL API key.  When omitted the ``DEEPL_API_KEY`` environment
        variable is used.  Free-tier keys end with ``:fx``.

    Raises
    ------
    ValueError
        At construction time if no API key is found.

    Examples
    --------
    Basic translation::

        from tools.services import serviceTranslate

        tool = serviceTranslate()

        # Call-style — returns ToolResult, never raises
        result = tool(input="Hello, world!", options={"target_lang": "DE"})
        if result:
            print(result.output)   # "Hallo, Welt!"
        else:
            print("Error:", result.error)

        # Direct run — returns str, raises on error
        translated = tool.run(
            input="Good morning",
            options={"target_lang": "RU"},
        )

    Formal German with contextual hint::

        tool.run(
            input="Can you help me with this?",
            options={
                "target_lang": "DE",
                "formality":   "more",
                "context":     "Customer support email to a corporate client",
            },
        )

    Rephrase / improve text style::

        improved = tool.rephrase(
            input="We are pleased to inform you of our decision.",
            options={"writing_style": "business", "tone": "confident"},
        )
    """

    name        = "serviceTranslate"
    description = (
        "Translate or rephrase text using the DeepL API. "
        "Use run() for translation, rephrase() to improve style."
    )
    parameters  = {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "Text to translate or rephrase.",
            },
            "options": {
                "type": "object",
                "description": "Translation options.",
                "properties": {
                    "target_lang": {
                        "type": "string",
                        "description": (
                            "Target language code, e.g. DE, FR, RU, ZH, EN-US. "
                            "Required for translation."
                        ),
                    },
                    "source_lang": {
                        "type": "string",
                        "description": "Source language code. Auto-detected when omitted.",
                    },
                    "formality": {
                        "type": "string",
                        "enum": ["default", "more", "less", "prefer_more", "prefer_less"],
                        "description": "Translation formality/tone.",
                    },
                    "context": {
                        "type": "string",
                        "description": (
                            "Contextual hint to improve translation accuracy "
                            "(not translated itself)."
                        ),
                    },
                    "preserve_formatting": {
                        "type": "boolean",
                        "description": "Preserve original whitespace and punctuation. Default: false.",
                    },
                },
                "required": ["target_lang"],
            },
        },
        "required": ["input"],
    }

    # ------------------------------------------------------------------

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get(_ENV_KEY)
        if not key:
            raise ValueError(
                f"No DeepL API key found.  Pass api_key= or set the "
                f"{_ENV_KEY} environment variable."
            )
        self._api_key = key
        self._http    = urllib3.PoolManager()

    # ------------------------------------------------------------------

    def run(
        self,
        input:   str,
        options: dict | None = None,
    ) -> str:
        """
        Translate *input* to the target language and return the translated string.

        Parameters
        ----------
        input : str
            Text to translate (UTF-8 plain text).
        options : dict | None, optional
            Translation options:
              ``target_lang``         — target language code (e.g. DE, FR, RU, ZH).
                                        Required.
              ``source_lang``         — source language code. Auto-detected if omitted.
              ``formality``           — tone: default | more | less |
                                        prefer_more | prefer_less.
              ``context``             — contextual hint (not translated itself) that
                                        improves translation accuracy.
              ``preserve_formatting`` — preserve original whitespace and punctuation.

        Returns
        -------
        str
            Translated text.

        Raises
        ------
        ValueError
            When ``target_lang`` is not provided in options.
        RuntimeError
            On any non-2xx HTTP response from the DeepL API.
        """
        opts                = options or {}
        target_lang         = opts.get("target_lang")
        source_lang         = opts.get("source_lang")
        formality           = opts.get("formality")
        context             = opts.get("context")
        preserve_formatting = opts.get("preserve_formatting", False)

        if not target_lang:
            raise ValueError(
                "target_lang is required for translation.  "
                "Pass it in options, e.g. options={'target_lang': 'DE'}."
            )

        payload: dict = {
            "text":        [input],
            "target_lang": target_lang.upper(),
        }

        if source_lang:
            payload["source_lang"] = source_lang.upper()
        if formality:
            payload["formality"] = formality
        if context:
            payload["context"] = context
        if preserve_formatting:
            payload["preserve_formatting"] = True

        headers = {
            "Authorization": f"DeepL-Auth-Key {self._api_key}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        }

        response = self._http.request(
            "POST",
            f"{_api_base(self._api_key)}/translate",
            body    = json.dumps(payload).encode("utf-8"),
            headers = headers,
            timeout = urllib3.Timeout(connect=10, read=30),
        )

        raw = response.data.decode("utf-8", errors="replace")
        if not (200 <= response.status < 300):
            raise RuntimeError(
                f"DeepL Translate API error [{response.status}]: {raw}"
            )

        return json.loads(raw)["translations"][0]["text"]

    def rephrase(
        self,
        input:   str,
        options: dict | None = None,
    ) -> str:
        """
        Rephrase and improve *input* using the DeepL Write API.

        Rewrites text for clarity, grammar, and style while preserving the
        original meaning.  Optionally applies a writing style and tone.

        Supported languages: de, en / en-GB / en-US, es, fr, it, ja, ko,
        pt / pt-BR / pt-PT, zh / zh-Hans.

        Parameters
        ----------
        input : str
            Text to improve (UTF-8 plain text).
        options : dict | None, optional
            Rephrasing options:
              ``target_lang``   — language code for the output (e.g. en, de, fr).
                                  Auto-detected from the input when omitted.
              ``writing_style`` — academic | business | casual | default | simple
                                  (and prefer_* variants).
              ``tone``          — confident | default | diplomatic | enthusiastic |
                                  friendly (and prefer_* variants).

        Returns
        -------
        str
            Rephrased / improved text.

        Raises
        ------
        RuntimeError
            On any non-2xx HTTP response from the DeepL API.
        """
        opts          = options or {}
        target_lang   = opts.get("target_lang")
        writing_style = opts.get("writing_style")
        tone          = opts.get("tone")

        payload: dict = {"text": [input]}

        if target_lang:
            payload["target_lang"] = target_lang.lower()
        if writing_style:
            payload["writing_style"] = writing_style
        if tone:
            payload["tone"] = tone

        headers = {
            "Authorization": f"DeepL-Auth-Key {self._api_key}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        }

        response = self._http.request(
            "POST",
            f"{_api_base(self._api_key)}/write/rephrase",
            body    = json.dumps(payload).encode("utf-8"),
            headers = headers,
            timeout = urllib3.Timeout(connect=10, read=30),
        )

        raw = response.data.decode("utf-8", errors="replace")
        if not (200 <= response.status < 300):
            raise RuntimeError(
                f"DeepL Rephrase API error [{response.status}]: {raw}"
            )

        return json.loads(raw)["improvements"][0]["text"]

    def __repr__(self) -> str:
        return f"serviceTranslate(name={self.name!r})"
