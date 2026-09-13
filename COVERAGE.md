# Coverage

The floor is **77.9%**, set in `pyproject.toml`
(`[tool.coverage.report] fail_under`). Measured 2026-09-13 at 77.07% on Python
3.10 and 3.14 alike, and raised the same day when the `Pool` and `Chain`
contract tests brought it to 77.93% on both. It only rises.

A number on its own invites the wrong game — raising it by testing whatever is
easy. So the number is what is left after decisions, and the decisions are
here: **every module below 60% is listed with a kind and a reason**, and
`scripts/coverage_register.py` checks the list both ways in CI. A gap with no
entry fails; an entry whose module has climbed past 60% also fails, because a
register that only grows stops being read.

Kinds:

- **external** — the uncovered lines are the request path to a vendor or a
  server. Construction, schema and validation are covered offline.
- **optional-dependency** — the uncovered lines need a package that is not a
  dependency of the library and is not installed in CI.
- **legacy** — exported, but unreachable from the 2.x primitives.
- **debt** — testable offline and not yet tested. This is the list to shrink.

Measured 2026-09-13, 32 modules holding 1,086 of the 1,694 uncovered
statements.

| Module | Kind | Reason |
|---|---|---|
| `yait_aichain/tools/search/perplexity.py` | external | request and result formatting need the Perplexity Search API |
| `yait_aichain/tools/search/serp.py` | external | request and per-engine result parsing need SerpAPI |
| `yait_aichain/tools/search/openai.py` | external | request and answer/citation parsing need the OpenAI Responses API |
| `yait_aichain/tools/search/brave.py` | external | request and result formatting need the Brave Search API |
| `yait_aichain/tools/convert/to_text.py` | external | transcription paths call four vendors' speech APIs |
| `yait_aichain/tools/convert/to_speech.py` | external | synthesis paths call four vendors' speech APIs |
| `yait_aichain/tools/embedding/_base.py` | external | the shared request path of every embedding provider |
| `yait_aichain/tools/embedding/__init__.py` | external | the factory resolves a provider client that needs its API |
| `yait_aichain/tools/embedding/_openai.py` | external | request shaping for the OpenAI embeddings API |
| `yait_aichain/tools/embedding/_cohere.py` | external | request shaping for the Cohere embeddings API |
| `yait_aichain/tools/embedding/_voyage.py` | external | request shaping for the Voyage embeddings API |
| `yait_aichain/tools/reranking/_base.py` | external | the shared request path of every reranker |
| `yait_aichain/tools/reranking/__init__.py` | external | the factory resolves a provider client that needs its API |
| `yait_aichain/tools/reranking/_cohere.py` | external | request shaping for the Cohere rerank API |
| `yait_aichain/tools/reranking/_voyage.py` | external | request shaping for the Voyage rerank API |
| `yait_aichain/tools/vectordb/providers/_chroma.py` | external | every operation needs a running Chroma server |
| `yait_aichain/tools/vectordb/providers/_pinecone.py` | external | every operation needs a Pinecone index |
| `yait_aichain/tools/vectordb/__init__.py` | external | `VectorDB` connects to a provider on construction |
| `yait_aichain/tools/convert/to_pdf.py` | optional-dependency | rendering needs `weasyprint`, which is not a dependency |
| `yait_aichain/tools/convert/to_html.py` | optional-dependency | rendering needs `mistletoe`, which is not a dependency |
| `yait_aichain/tools/convert/to_md.py` | optional-dependency | conversion needs `markitdown` and, for URLs, the network |
| `yait_aichain/agent/_memory.py` | legacy | `AgentMemory` is exported but no 2.x primitive uses it — the agent's state is its conversation; a removal candidate, which is a breaking change and needs its own release |
| `yait_aichain/tools/local/_browse.py` | debt | sandboxed and filesystem-only; a temporary directory is all a test needs |
| `yait_aichain/tools/local/_read.py` | debt | sandboxed and filesystem-only |
| `yait_aichain/tools/local/_write.py` | debt | sandboxed and filesystem-only |
| `yait_aichain/tools/local/_run.py` | debt | runs a subprocess inside the sandbox; testable with `echo` |
| `yait_aichain/tools/local/_base.py` | debt | the sandbox check itself — path traversal and symlinks — deserves tests more than most |
| `yait_aichain/tools/vectordb/_query.py` | debt | wraps a `VectorStore`; an in-memory fake store is enough |
| `yait_aichain/tools/vectordb/_upsert.py` | debt | wraps a `VectorStore`; an in-memory fake store is enough |
| `yait_aichain/tools/vectordb/_fetch.py` | debt | wraps a `VectorStore`; an in-memory fake store is enough |
| `yait_aichain/tools/rest_api.py` | debt | the request and auth paths go through `self._http`, which can be replaced |
| `yait_aichain/tools/convert/__init__.py` | debt | the `TTS`/`STT` dispatch picks a class by name; no network is needed to test the choice |
