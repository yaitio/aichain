"""
``RerankCohere``: the Cohere Rerank v2 API (``POST /v2/rerank``). Cohere bills
in search units, which ``total_tokens`` carries. See docs/tools/retrieval.md.
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

        from yait_aichain.tools import rerank

        reranker = rerank("cohere", model="rerank-v3.5")
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
        }
        if top_n is not None:
            body["top_n"] = top_n

        data = self._post(_BASE_URL, body)

        results = self._ranked(data.get("results", []), documents)
        meta         = data.get("meta", {})
        billed       = meta.get("billed_units", {})
        total_tokens = billed.get("search_units")   # Cohere bills in search_units
        metadata     = {"billed_units": billed} if billed else {}

        return results, total_tokens, metadata
