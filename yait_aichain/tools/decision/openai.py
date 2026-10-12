"""
tools.decision.openai — decisionOpenAI
======================================

OpenAI's Decisions API behind the common :class:`~tools.decision.Decision`
format.

    POST https://api.openai.com/v1/decisions
         {"model": "gpt-6-luna", "input": ..., "questions": [{type, name, ...}]}
       → {"model": ..., "answers": [{type, name, ...}], "usage": {...}}

Docs: https://developers.openai.com/api/docs/guides/decisions — read and
checked live on 2026-10-08, when the API was in public beta with
``gpt-6-luna`` the only model.

Translated both ways: questions travel as a list carrying their ``name``,
``choices`` as ``[{value, description}]``, ``levels`` as ``[{label,
description}]``; answers come back as a list and are keyed by name again, with
their probabilities keyed by option and by level label.

This provider takes **images** beside the text, as base64 data URLs only —
hosted URLs and file ids are refused by the endpoint. A dict or a list state
from code is sent as JSON text: the endpoint takes text and messages, not
structure.

Environment variable
--------------------
``OPENAI_API_KEY``
"""

from __future__ import annotations

import json

from ._base import Decision, _level

_ENDPOINT = "https://api.openai.com/v1/decisions"


def _wire_questions(questions: dict) -> list:
    out = []
    for name, q in questions.items():
        w = {"type": q["type"], "name": name, "instructions": q["instructions"]}
        if q["type"] == "choice":
            w["choices"] = [{"value": v, **({"description": d} if d else {})}
                            for v, d in q["choices"].items()]
        elif q["type"] == "score":
            w["levels"] = [{"label": lb, **({"description": d} if d else {})}
                           for lb, d in map(_level, q["levels"])]
        out.append(w)
    return out


def _answer(q: dict, a: dict) -> dict:
    if q["type"] == "predicate":
        return {"type": "predicate", "probability": a.get("probability")}
    probs = a.get("probabilities") or []
    if q["type"] == "choice":
        by = {p.get("value"): p.get("probability") for p in probs}
        return {"type": "choice", "choice": a.get("choice"), "confidence": a.get("confidence"),
                "probabilities": {k: by.get(k) for k in q["choices"]}}
    by = {p.get("value"): p.get("probability") for p in probs}
    labels = [_level(x)[0] for x in q["levels"]]
    return {"type": "score", "score": a.get("score"), "confidence": a.get("confidence"),
            "probabilities": {lb: by.get(i) for i, lb in enumerate(labels)}}


class decisionOpenAI(Decision):
    """
    Typed questions about a text — and images — answered with probabilities,
    by OpenAI's Decisions API.

    ``model`` defaults to ``"gpt-6-luna"``, the only one the API serves. Input
    costs $0.10 per million tokens; output, cache reads and cache writes are
    free.

    Examples
    --------
    ::

        from yait_aichain.tools import decisionOpenAI

        inspect = decisionOpenAI(name="inspect", questions={
            "damaged": {"type": "predicate",
                        "instructions": "Is there visible damage — a crack, tear or dent?"},
        })
        inspect.run(state="Inspect the product in this photo.", images=["product.png"])
    """

    name                   = "decisionOpenAI"
    _ENV_KEY               = "OPENAI_API_KEY"
    _PROVIDER              = "OpenAI"
    _DEFAULT_MODEL         = "gpt-6-luna"
    #: "input costs $0.10 per 1M tokens. You pay only for input tokens: there
    #: are no cache-read, cache-write, or output-token charges" — the Decisions
    #: guide, read 2026-10-08. Regional and long-context multipliers are not
    #: applied here.
    _PRICE_PER_INPUT_TOKEN = 0.10 / 1_000_000
    _PRICE_SOURCE          = "the Decisions guide, read 2026-10-08"
    images                 = True

    _ENDPOINT              = _ENDPOINT
    _API_NAME              = "OpenAI Decisions API"
    _answer                = staticmethod(_answer)

    def _body(self, state, images):
        text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        given = text if not images else [{"role": "user", "content": [
            {"type": "input_text", "text": text},
            *({"type": "input_image", "image_url": url} for url in images)]}]
        return {"model": self.model, "input": given, "questions": _wire_questions(self.questions)}

    def _raw_answers(self, data):
        answers = data.get("answers")
        return {a.get("name"): a for a in answers} if isinstance(answers, list) else None
