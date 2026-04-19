"""
tools.search.openai — searchOpenAI
====================================

Wraps the OpenAI built-in web search capability behind the standard
:class:`~tools.search.Search` interface.

Endpoint  : POST https://api.openai.com/v1/responses
Auth      : ``Authorization: Bearer <api_key>``
Docs      : https://platform.openai.com/docs/guides/tools-web-search

How it works
------------
Unlike :class:`~tools.search.searchBrave` or
:class:`~tools.search.searchPerplexity` which return raw result lists,
this tool asks an OpenAI model to perform a live web search **and
synthesize a complete answer** in a single call.  The model decides which
sources to consult and cites them inline.

The tool returns a plain-text string with two sections::

    [Answer]
    The latest fusion energy developments include NIF achieving… [1]
    Commonwealth Fusion expects a commercial reactor by 2030… [2]

    [Sources]
    [1] NIF Achieves Ignition — Science Daily
        https://www.sciencedaily.com/releases/…
    [2] Commonwealth Fusion Systems — Company Blog
        https://cfs.energy/news/…

Model compatibility
-------------------
The ``web_search_preview`` tool is supported on:
``gpt-4o``, ``gpt-4.1``, ``gpt-4.1-mini``, ``o3``, ``o4-mini``.
Not supported on ``gpt-4.1-nano``.  Default is ``gpt-4o``.

Environment variable
---------------------
``OPENAI_API_KEY`` — standard OpenAI secret key.
"""

from __future__ import annotations

import json
import os

import urllib3

from ._base import Search

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENDPOINT      = "https://api.openai.com/v1/responses"
_ENV_KEY       = "OPENAI_API_KEY"
_DEFAULT_MODEL = "gpt-4o"


# ---------------------------------------------------------------------------
# searchOpenAI
# ---------------------------------------------------------------------------

class searchOpenAI(Search):
    """
    Web search powered by OpenAI's built-in ``web_search_preview`` tool.

    The model performs a live web search and synthesizes a complete answer
    with inline citations in a single API call — no follow-up URL fetch
    step required.

    Parameters
    ----------
    api_key : str | None, optional
        OpenAI secret key (``sk-...``).  When omitted the value of the
        ``OPENAI_API_KEY`` environment variable is used.
    model : str, optional
        OpenAI model to use for the search.  Defaults to ``"gpt-4o"``.
        Must support the ``web_search_preview`` tool.

    Raises
    ------
    ValueError
        At construction time if no API key is found.

    Examples
    --------
    Basic search::

        from tools.search import searchOpenAI

        tool   = searchOpenAI()
        result = tool(input="nuclear fusion breakthroughs 2025")

        if result:
            print(result.output)   # synthesized answer + numbered sources
        else:
            print("Error:", result.error)

    With context size and domain filter::

        text = tool.run(
            input   = "GPT-5 benchmark results",
            options = {
                "context_size": "high",
                "domains":      ["openai.com", "arxiv.org"],
            },
        )
        print(text)

    Using a different model::

        tool = searchOpenAI(model="gpt-4.1")
        text = tool.run(input="latest AI safety research 2025")
    """

    name        = "searchOpenAI"
    description = (
        "Search the web using OpenAI's built-in web search. "
        "Returns a synthesized answer with inline citations."
    )
    parameters  = {
        "type": "object",
        "properties": {
            "input": {
                "type":        "string",
                "description": (
                    "The question or search query to answer using live web search."
                ),
            },
            "options": {
                "type":        "object",
                "description": "Optional search filters and configuration.",
                "properties": {
                    "context_size": {
                        "type":        "string",
                        "enum":        ["low", "medium", "high"],
                        "description": (
                            "How much web context to gather before answering.  "
                            "'low' is fast and cheap; 'high' is thorough but uses more "
                            "tokens.  Defaults to 'medium'."
                        ),
                    },
                    "domains": {
                        "type":        "array",
                        "items":       {"type": "string"},
                        "description": (
                            "Restrict the search to these domains only "
                            "(e.g. [\"openai.com\", \"arxiv.org\"]).  "
                            "Subdomains are automatically included."
                        ),
                    },
                    "country": {
                        "type":        "string",
                        "description": (
                            "Two-letter ISO 3166-1 country code to bias results "
                            "(e.g. 'US', 'GB').  Part of the approximate user location."
                        ),
                    },
                    "timezone": {
                        "type":        "string",
                        "description": (
                            "IANA timezone string to help with time-sensitive queries "
                            "(e.g. 'America/New_York').  Part of the approximate user "
                            "location."
                        ),
                    },
                    "model": {
                        "type":        "string",
                        "description": (
                            "Override the OpenAI model for this call.  "
                            "Defaults to the model passed at construction time."
                        ),
                    },
                },
            },
        },
        "required": ["input"],
    }

    # ------------------------------------------------------------------

    def __init__(
        self,
        api_key: str | None = None,
        model:   str        = _DEFAULT_MODEL,
    ) -> None:
        key = api_key or os.environ.get(_ENV_KEY)
        if not key:
            raise ValueError(
                f"No OpenAI API key found.  Pass api_key= or set the "
                f"{_ENV_KEY} environment variable."
            )
        self._api_key = key
        self._model   = model
        self._http    = urllib3.PoolManager()

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def run(self, input: str, options: "dict | None" = None) -> str:
        """
        Ask an OpenAI model to search the web and return a synthesized answer.

        Parameters
        ----------
        input : str
            Question or search query (required).
        options : dict | None
            Optional configuration keys:
              ``context_size`` — ``"low"`` / ``"medium"`` / ``"high"``.
                                 Default ``"medium"``.
              ``domains``      — restrict search to these domains.
              ``country``      — ISO 3166-1 alpha-2 country code.
              ``timezone``     — IANA timezone string for time-sensitive queries.
              ``model``        — override the OpenAI model for this call.

        Returns
        -------
        str
            Two-section plain text::

                [Answer]
                Synthesized answer text with [N] inline citation markers…

                [Sources]
                [1] Page Title — https://url
                [2] …

        Raises
        ------
        RuntimeError
            On any non-2xx HTTP response from the OpenAI API.
        """
        opts = options or {}

        search_context_size = opts.get("context_size", "medium")
        allowed_domains     = opts.get("domains")
        country             = opts.get("country")
        timezone            = opts.get("timezone")
        model               = opts.get("model", self._model)

        # ── Build web_search tool config ──────────────────────────────
        search_tool: dict = {
            "type":                "web_search_preview",
            "search_context_size": search_context_size,
        }

        if country or timezone:
            location: dict = {"type": "approximate"}
            if country:  location["country"]  = country
            if timezone: location["timezone"] = timezone
            search_tool["user_location"] = location

        if allowed_domains:
            search_tool["filters"] = {"allowed_domains": allowed_domains}

        # ── Build Responses API request body ─────────────────────────
        body: dict = {
            "model": model,
            "input": input,
            "tools": [search_tool],
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        }

        # ── Send request ──────────────────────────────────────────────
        response = self._http.request(
            "POST",
            _ENDPOINT,
            body    = json.dumps(body).encode("utf-8"),
            headers = headers,
        )

        raw = response.data.decode("utf-8", errors="replace")

        if not (200 <= response.status < 300):
            raise RuntimeError(
                f"OpenAI Web Search API error [{response.status}]: {raw}"
            )

        data = json.loads(raw)

        # ── Extract answer and citations ──────────────────────────────
        answer_text = ""
        annotations: list[dict] = []

        for item in data.get("output", []):
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        answer_text = part.get("text", "")
                        annotations = part.get("annotations", [])
                        break
                break

        if not answer_text:
            return "(no answer returned)"

        # ── Format output ─────────────────────────────────────────────
        lines = ["[Answer]", answer_text.strip(), ""]

        url_citations = [
            a for a in annotations
            if a.get("type") == "url_citation"
        ]

        if url_citations:
            lines.append("[Sources]")
            seen: set[str] = set()
            idx = 1
            for ann in url_citations:
                url = ann.get("url", "")
                if url in seen:
                    continue
                seen.add(url)
                title = ann.get("title", "(no title)")
                lines.append(f"[{idx}] {title}")
                lines.append(f"    {url}")
                idx += 1

        return "\n".join(lines)
