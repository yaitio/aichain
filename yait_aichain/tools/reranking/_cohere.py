"""
tools.reranking._cohere — RerankCohere
=======================================

Wraps the Cohere Rerank API v2 (``POST /v2/rerank``).

Supported models
----------------
  rerank-v3.5                  Latest; multilingual; best quality
  rerank-english-v3.0          English-only; strong baseline
  rerank-multilingual-v3.0     Multilingual; 100+ languages

API reference
-------------
POST https://api.cohere.com/v2/rerank

Request body
  model            : str
  query            : str
  documents        : list[str]
  top_n            : int | None
  return_documents : true

Response
  results[].index            — original document index
  results[].relevance_score  — float, higher = more relevant
  results[].document.text    — document text (when return_documents=true)
  meta.billed_units.search_units — billing

Environment variable
--------------------
COHERE_API_KEY
"""

from __future__ import annotations

from ._base import RerankBase

_BASE_URL = "https://api.cohere.com/v2/rerank"


class RerankCohere(RerankBase):
    """
    Reranker for the Cohere Rerank API (v2).

    Parameters
    ----------
    model : str
        Model identifier, e.g. ``"rerank-v3.5"``.
    api_key : str | None, optional
        Override the ``COHERE_API_KEY`` environment variable.

    Examples
    --------
    ::

        from tools.reranking import Reranker

        reranker = Reranker("cohere/rerank-v3.5")
        result   = reranker.rerank(
            "how does KV caching work?",
            ["LLMs use KV cache …", "Python is a language …", "Attention is …"],
            top_n=2,
        )
        for item in result:
            print(f"{item['score']:.4f}  {item['text'][:60]}")
    """

    provider = "cohere"
    _ENV_KEY = "COHERE_API_KEY"

    def _rerank_request(
        self,
        query:     str,
        documents: list[str],
        top_n:     int | None,
    ) -> tuple[list[dict], int | None, dict]:
        body: dict = {
            "model":            self.model,
            "query":            query,
            "documents":        documents,
            "return_documents": True,
        }
        if top_n is not None:
            body["top_n"] = top_n

        data = self._post(_BASE_URL, body)

        results: list[dict] = []
        for item in data.get("results", []):
            results.append({
                "index": item["index"],
                "text":  (item.get("document") or {}).get("text", documents[item["index"]]),
                "score": item["relevance_score"],
            })
        # Already sorted by Cohere (descending score)

        meta         = data.get("meta", {})
        billed       = meta.get("billed_units", {})
        total_tokens = billed.get("search_units")   # Cohere bills in search_units
        metadata     = {"billed_units": billed} if billed else {}

        return results, total_tokens, metadata
