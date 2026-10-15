"""TypeSafe AI's Jev (``POST https://api.typesafe.ai/v1/systemone``) behind the
common :class:`~tools.decision.Decision` format. ``predicate`` is Jev's
``noul``, ``choices`` and ``levels`` its ``criteria`` (a map for a choice, a
list for a score), and a score's probabilities, keyed by level index on the
wire, come back keyed by level label. A dict or list state travels as
structure; no images. Key: ``TYPESAFE_API_KEY``. See docs/tools/decide.md.
"""

from __future__ import annotations


from ._base import Decision, _level

_ENDPOINT = "https://api.typesafe.ai/v1/systemone"



def _wire_questions(questions: dict) -> dict:
    out = {}
    for name, q in questions.items():
        w = {"type": "noul" if q["type"] == "predicate" else q["type"],
             "instructions": q["instructions"]}
        if q["type"] == "choice":
            w["criteria"] = dict(q["choices"])
        elif q["type"] == "score":
            # Jev takes one string per level; a description rides after its label.
            w["criteria"] = [f"{lb}: {d}" if d else lb for lb, d in map(_level, q["levels"])]
        out[name] = w
    return out


def _answer(q: dict, a: dict) -> dict:
    if q["type"] == "predicate":
        return {"type": "predicate", "probability": a.get("noul")}
    if q["type"] == "choice":
        return {"type": "choice", "choice": a.get("choice"), "confidence": a.get("confidence"),
                "probabilities": {k: a.get("probabilities", {}).get(k) for k in q["choices"]}}
    labels = [_level(x)[0] for x in q["levels"]]
    probs = a.get("probabilities") or {}
    return {"type": "score", "score": a.get("score"), "confidence": a.get("confidence"),
            "probabilities": {lb: probs.get(str(i)) for i, lb in enumerate(labels)}}


class decisionTypeSafe(Decision):
    """
    Typed questions about a text, answered with probabilities, by TypeSafe's Jev.

    ``model`` is ``"jev-latest"`` (default), ``"jev-preview"`` or a versioned
    id such as ``"jev-1.13.0"`` — pin one when thresholds were tuned against
    it, because an alias moves when a release ships. The answer names the
    version that answered. Input costs $0.042 per million tokens; output is
    free.

    Examples
    --------
    ::

        from yait_aichain.tools import decisionTypeSafe

        triage = decisionTypeSafe(name="triage", questions={
            "urgent": {"type": "predicate", "instructions": "Does this convey urgency?"},
            "team":   {"type": "choice", "instructions": "Which team should handle it?",
                       "choices": {"billing": "Payments, refunds", "technical": "Bugs, outages"}},
        })
        triage.run(state="Help! My payouts have been failing for 3 days.")
    """

    name                   = "decisionTypeSafe"
    _ENV_KEY               = "TYPESAFE_API_KEY"
    _PROVIDER              = "TypeSafe"
    _DEFAULT_MODEL         = "jev-latest"
    #: "$42 / $0.042 per Btok / per Mtok … Charged per input token. Output
    #: tokens are free" — docs.typesafe.ai/models, read 2026-10-02, jev-1.13.0.
    _PRICE_PER_INPUT_TOKEN = 0.042 / 1_000_000
    _PRICE_SOURCE          = "docs.typesafe.ai/models, read 2026-10-02"
    images                 = False

    _ENDPOINT              = _ENDPOINT
    _API_NAME              = "TypeSafe API"
    _answer                = staticmethod(_answer)

    def _body(self, state, images):
        return {"state": state, "model": self.model, "questions": _wire_questions(self.questions)}

    def _raw_answers(self, data):
        answers = data.get("answers")
        return answers if isinstance(answers, dict) else None
