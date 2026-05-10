"""
tools.reranking._voyage — RerankVoyage
========================================

Wraps the Voyage AI Rerank API (``POST /v1/rerank``).

Supported models
----------------
  rerank-2       Latest; strong across all domains; default choice
  rerank-2-lite  Faster and cheaper; good for high-throughput pipelines

API reference
-------------
POST https://api.voyageai.com/v1/rerank

Request body
  model            : str
  query            : str
  documents        : list[str]
  top_k            : int | None    (Voyage uses top_k, not top_n)
  return_documents : true

Response
  data[].index           — original document index
  data[].relevance_score — float, higher = more relevant
  data[].document        — document text string (when return_documents=true)
  usage.total_tokens     — token count

Environment variable
--------------------
VOYAGE_API_KEY
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

        from tools.reranking import Reranker

        reranker = Reranker("voyage/rerank-2")
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
            "return_documents": True,
        }
        if top_n is not None:
            body["top_k"] = top_n   # Voyage uses top_k

        data = self._post(_BASE_URL, body)

        results: list[dict] = []
        for item in data.get("data", []):
            # Voyage returns document as a plain string when return_documents=True
            doc_text = item.get("document", documents[item["index"]])
            if isinstance(doc_text, dict):
                doc_text = doc_text.get("text", documents[item["index"]])
            results.append({
                "index": item["index"],
                "text":  doc_text,
                "score": item["relevance_score"],
            })
        # Sort descending by score (Voyage already sorts but be safe)
        results.sort(key=lambda x: x["score"], reverse=True)

        usage        = data.get("usage", {})
        total_tokens = usage.get("total_tokens")

        return results, total_tokens, {}
