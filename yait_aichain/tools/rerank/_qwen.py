"""
``RerankQwen``: the DashScope Text-Rerank API; the host follows ``region`` or
``DASHSCOPE_REGION``, as for ``EmbeddingQwen``. See docs/tools/retrieval.md.
"""

from __future__ import annotations


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

        from yait_aichain.tools import rerank

        reranker = rerank("qwen", model="gte-rerank")
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
        from ...clients._families.qwen import resolve_qwen_base_url

        base_url = resolve_qwen_base_url(self._region)
        url      = base_url + _RERANK_PATH

        params: dict = {}
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
        results = self._ranked(output.get("results", []), documents)
        usage        = data.get("usage", {})
        total_tokens = usage.get("total_tokens")

        return results, total_tokens, {}
