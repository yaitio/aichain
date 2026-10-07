"""
tools.decision.typesafe — decisionTypeSafe
==========================================

TypeSafe AI's Jev behind the common :class:`~tools.decision.Decision` format.

    POST https://api.typesafe.ai/v1/systemone
         {"state": ..., "model": "jev-latest", "questions": {id: Question}}
       → {"model": "jev-1.13.0", "answers": {id: Answer}, "usage": {...}}

Translated both ways: ``predicate`` is Jev's ``noul``, ``choices`` and
``levels`` are its ``criteria`` (a map for a choice, a list for a score), and a
score's probabilities, keyed by level index on the wire, come back keyed by
level label like every other provider's.

Jev takes structured state: a dict or a list from code travels as structure.
It takes no images.

Environment variable
--------------------
``TYPESAFE_API_KEY`` — from https://console.typesafe.ai
"""

from __future__ import annotations

import json

from ...clients._constants import DEFAULT_RETRIES
from ._base import Decision, _level

_ENDPOINT = "https://api.typesafe.ai/v1/systemone"

#: 529 is TypeSafe's "temporarily overloaded" — refused before inference, like
#: the 429 and 503 every client already retries, so a POST is safe to repeat.
_RETRIES = DEFAULT_RETRIES.new(
    status_forcelist=set(DEFAULT_RETRIES.status_forcelist) | {529})


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
    images                 = False
    _RETRIES               = _RETRIES

    def _ask(self, state, images):
        body = {"state": state, "model": self.model, "questions": _wire_questions(self.questions)}
        response = self._http.request(
            "POST", _ENDPOINT, body=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"})
        raw = response.data.decode("utf-8", errors="replace")
        if not 200 <= response.status < 300:
            raise RuntimeError(f"TypeSafe API error [{response.status}]: {raw}")
        data = json.loads(raw)
        answers = data.get("answers")
        if not isinstance(answers, dict):
            raise RuntimeError(f"TypeSafe returned no answers: {raw}")
        usage = data.get("usage") or {}
        return ({n: _answer(q, answers.get(n) or {}) for n, q in self.questions.items()},
                data.get("model", self.model),
                {"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")})
