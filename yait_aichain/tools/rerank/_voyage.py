"""
``RerankVoyage``: the Voyage AI Rerank API (``POST /v1/rerank``), where
``top_n`` is sent as ``top_k``. See docs/tools/retrieval.md.
"""

from __future__ import annotations

from ._base import RerankBase

_BASE_URL = "https://api.voyageai.com/v1/rerank"


class RerankVoyage(RerankBase):
    """
    Reranker for the Voyage AI Rerank API.

    Parameters
    ----------
    model : str
        Model identifier, e.g. ``"rerank-2"``.
    api_key : str | None, optional
        Override the ``VOYAGE_API_KEY`` environment variable.

    Examples
    --------
    ::

        from yait_aichain.tools import rerank

        reranker = rerank("voyage", model="rerank-2")
        result   = reranker.rerank(
            "transformer architecture",
            ["Attention is all you need …", "Python syntax …", "BERT uses …"],
            top_n=2,
        )
        print(result.texts())
    """

    provider = "voyage"
    _ENV_KEY = "VOYAGE_API_KEY"

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
            body["top_k"] = top_n   # Voyage uses top_k

        data = self._post(_BASE_URL, body)

        results = self._ranked(data.get("data", []), documents)
        usage        = data.get("usage", {})
        total_tokens = usage.get("total_tokens")

        return results, total_tokens, {}
