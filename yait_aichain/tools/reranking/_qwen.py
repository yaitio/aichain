"""
tools.reranking._qwen — RerankQwen
=====================================

Wraps the Alibaba DashScope Text-Rerank API.

Supported models
----------------
  gte-rerank      Latest GTE reranker; strong multilingual performance
  gte-rerank-v2   Previous generation; still solid

API reference
-------------
POST {base_url}/api/v1/services/rerank/text-rerank/text-rerank

Request body
  model       : str
  input
    query     : str
    documents : list[str]
  parameters
    top_n            : int | None
    return_documents : true

Response
  output.results[].index           — original document index
  output.results[].relevance_score — float, higher = more relevant
  output.results[].document.text   — document text
  usage.total_tokens               — token count

Region / base URL
-----------------
Same region logic as EmbeddingQwen (``DASHSCOPE_REGION`` env var):
  ap (default) — https://dashscope-intl.aliyuncs.com
  us           — https://dashscope-us.aliyuncs.com
  cn           — https://dashscope.aliyuncs.com
  hk           — https://cn-hongkong.dashscope.aliyuncs.com

Environment variables
---------------------
DASHSCOPE_API_KEY
DASHSCOPE_REGION  (optional; default: ap)
"""

from __future__ import annotations

import os

from ._base import RerankBase

_RERANK_PATH = "/api/v1/services/rerank/text-rerank/text-rerank"


class RerankQwen(RerankBase):
    """
    Reranker for the Alibaba DashScope Text-Rerank API.

    Parameters
    ----------
    model : str
        Model identifier, e.g. ``"gte-rerank"``.
    api_key : str | None, optional
        Override the ``DASHSCOPE_API_KEY`` environment variable.
    region : str | None, optional
        Region selector: ``"ap"`` (default), ``"us"``, ``"cn"``, ``"hk"``.
        Overrides the ``DASHSCOPE_REGION`` environment variable.

    Examples
    --------
    ::

        from tools.reranking import Reranker

        reranker = Reranker("qwen/gte-rerank")
        result   = reranker.rerank(
            "machine learning basics",
            ["ML is a subset of AI …", "Python syntax …", "Neural networks …"],
            top_n=2,
        )
        print(result.scores())
    """

    provider = "qwen"
    _ENV_KEY = "DASHSCOPE_API_KEY"

    def __init__(
        self,
        model:   str,
        *,
        api_key: str | None = None,
        region:  str | None = None,
        **kwargs,
    ) -> None:
        self._region = region
        super().__init__(model, api_key=api_key, **kwargs)

    def _rerank_request(
        self,
        query:     str,
        documents: list[str],
        top_n:     int | None,
    ) -> tuple[list[dict], int | None, dict]:
        from clients._qwen import resolve_qwen_base_url

        base_url = resolve_qwen_base_url(self._region)
        url      = base_url + _RERANK_PATH

        params: dict = {"return_documents": True}
        if top_n is not None:
            params["top_n"] = top_n

        body: dict = {
            "model": self.model,
            "input": {
                "query":     query,
                "documents": documents,
            },
            "parameters": params,
        }

        data = self._post(url, body)

        output  = data.get("output", {})
        results: list[dict] = []
        for item in output.get("results", []):
            doc  = item.get("document") or {}
            text = doc.get("text", documents[item["index"]])
            results.append({
                "index": item["index"],
                "text":  text,
                "score": item["relevance_score"],
            })
        # Sort descending by score
        results.sort(key=lambda x: x["score"], reverse=True)

        usage        = data.get("usage", {})
        total_tokens = usage.get("total_tokens")

        return results, total_tokens, {}
