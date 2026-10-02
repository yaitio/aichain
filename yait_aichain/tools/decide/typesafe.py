"""
tools.decide.typesafe — decideTypeSafe
======================================

TypeSafe AI's Jev as a tool: a text in, typed answers out — with a probability
for every option.

    POST https://api.typesafe.ai/v1/systemone
         {"state": ..., "model": "jev-latest", "questions": {id: Question}}
       → {"model": "jev-1.13.0", "answers": {id: Answer}, "usage": {...}}

Jev is not a chat model. It writes no text, calls no tools and holds no
conversation: it answers typed questions about a state — ``noul`` (the
probability a statement is true), ``choice`` (one option from a map, a
probability for each, ``confidence``) and ``score`` (a probability-weighted
value over ordered levels, ``confidence``). All of them in one call, in
parallel, against the same state.

**Why a tool and not a Model.** Every specialised model here that does not
converse — speech, transcription, embeddings, reranking, search — lives in
``tools/``. Jev is one more; added as a ``Model`` on 2026-10-02 it took changes
in ten places outside its own client, and an ``Agent`` could not use it at all.
As a tool it touches nothing else, and an agent gets the one thing Jev is for:
a calibrated decision it can act on, or decline to act on, by the number.

**The questions are fixed when the tool is made.** The agent passes only the
text to judge. Letting the model write the questions would let it write them
badly, and the reliability that is the reason to reach for Jev would go with
them. Name each tool by what it decides — ``decideTypeSafe(name="triage",
questions=...)`` — and the agent sees ``triage(state)``.

**Answers come back untouched.** A ``noul`` of 0.95 stays 0.95: the
probability is the product, and the threshold belongs to the caller, who
knows what a wrong answer costs.

**Money.** The answer carries ``usage`` and ``cost`` — input tokens only,
output is free. It is reported, not budgeted: ``max_cost`` is counted inside
``Skill``, and no tool in this library is held to it yet.

Environment variable
--------------------
``TYPESAFE_API_KEY`` — from https://console.typesafe.ai
"""

from __future__ import annotations

import json
import os

from ...clients._base import make_http
from ...clients._constants import DEFAULT_RETRIES, DEFAULT_TIMEOUT
from .._base import Tool

_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
_ENV_KEY  = "TYPESAFE_API_KEY"

#: USD per input token. "$42 / $0.042 per Btok / per Mtok … Charged per input
#: token. Output tokens are free" — docs.typesafe.ai/models, read 2026-10-02,
#: for jev-1.13.0, which both aliases pointed to that day.
_PRICE_PER_INPUT_TOKEN = 0.042 / 1_000_000

_QUESTION_TYPES     = ("noul", "choice", "score")
_MAX_CHOICE_OPTIONS = 255
_SCORE_LEVELS       = (2, 10)

#: 529 is TypeSafe's "temporarily overloaded" — refused before inference, like
#: the 429 and 503 every client already retries, so a POST is safe to repeat.
_RETRIES = DEFAULT_RETRIES.new(
    status_forcelist=set(DEFAULT_RETRIES.status_forcelist) | {529})


def _check_questions(questions) -> None:
    """Refuse a malformed question here, naming it, rather than as a 422."""
    if not isinstance(questions, dict) or not questions:
        raise ValueError(
            "decideTypeSafe needs at least one question, as a map of id → "
            "question, e.g. {'urgent': {'type': 'noul', 'instructions': "
            "'Is this urgent?'}}")
    for qid, q in questions.items():
        if not isinstance(q, dict):
            raise ValueError(f"question {qid!r} must be a mapping, got "
                             f"{type(q).__name__}")
        kind = q.get("type")
        if kind not in _QUESTION_TYPES:
            raise ValueError(f"question {qid!r} has type {kind!r}; it must be "
                             f"one of {', '.join(_QUESTION_TYPES)}")
        if not q.get("instructions"):
            raise ValueError(f"question {qid!r} has no instructions — that is "
                             "the question itself")
        crit = q.get("criteria")
        if kind == "choice":
            if not isinstance(crit, dict) or not crit:
                raise ValueError(f"choice {qid!r} needs criteria: a map of "
                                 "option → description (null when an option "
                                 "needs none)")
            if len(crit) > _MAX_CHOICE_OPTIONS:
                raise ValueError(f"choice {qid!r} has {len(crit)} options; the "
                                 f"API takes at most {_MAX_CHOICE_OPTIONS}")
        elif kind == "score":
            lo, hi = _SCORE_LEVELS
            if not isinstance(crit, list) or not lo <= len(crit) <= hi:
                raise ValueError(f"score {qid!r} needs criteria: an ordered "
                                 f"list of {lo} to {hi} level descriptions")


def _describe(questions: dict) -> str:
    """What the tool tells the model it will learn — one line per question."""
    lines = []
    for qid, q in questions.items():
        instr = q["instructions"]
        text = instr if isinstance(instr, str) else json.dumps(instr)
        if q["type"] == "noul":
            what = "probability that the answer is yes"
        elif q["type"] == "choice":
            what = "one of " + ", ".join(map(str, q["criteria"])) + ", with probabilities"
        else:
            what = f"a score from 0 to {len(q['criteria']) - 1} on: " + " < ".join(
                c if isinstance(c, str) else json.dumps(c) for c in q["criteria"])
        lines.append(f"- {qid}: {text} → {what}")
    return ("Judge a text against fixed questions and get calibrated answers "
            "back — probabilities, not prose. Pass the text as `state`.\n"
            + "\n".join(lines))


class decideTypeSafe(Tool):
    """
    Typed questions about a text, answered with probabilities, by TypeSafe's Jev.

    Parameters
    ----------
    questions : dict
        Map of id → question, in the API's own shape. Fixed for the life of
        the tool; the caller passes only the text.
    name : str, optional
        The name the model sees. Default ``"decideTypeSafe"``; name it by what
        it decides (``"triage"``, ``"is_duplicate"``).
    description : str, optional
        Default: written from the questions, so the model knows what it will
        learn by calling it.
    model : str
        ``"jev-latest"`` (default), ``"jev-preview"``, or a versioned id such
        as ``"jev-1.13.0"`` — pin one when confidence thresholds were tuned
        against it, because an alias moves when a release ships.
    api_key : str, optional
        Overrides ``TYPESAFE_API_KEY``.

    Examples
    --------
    ::

        from yait_aichain.tools import decideTypeSafe

        triage = decideTypeSafe(name="triage", questions={
            "urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
            "team":   {"type": "choice", "instructions": "Which team should handle it?",
                       "criteria": {"billing": "Payments, refunds",
                                    "technical": "Bugs, outages"}},
        })
        out = triage.run(state="Help! My payouts have been failing for 3 days.")
        out["answers"]["urgent"]["noul"]          # 0.95
        out["answers"]["team"]["choice"]          # "billing"
        out["answers"]["team"]["confidence"]      # 0.83

        Agent(model, tools=[triage])              # the agent calls triage(state)
    """

    name        = "decideTypeSafe"
    description = "Judge a text against fixed typed questions; returns probabilities."
    # It reads the text it is given and changes nothing anywhere.
    risk        = "read"
    parameters  = {
        "type": "object",
        "properties": {
            "state": {
                "type": "string",
                "description": "The text to judge — a message, a ticket, a record.",
            },
        },
        "required": ["state"],
    }

    def __init__(self, questions: dict, *, name: "str | None" = None,
                 description: "str | None" = None, model: str = "jev-latest",
                 api_key: "str | None" = None) -> None:
        _check_questions(questions)
        key = api_key or os.environ.get(_ENV_KEY)
        if not key:
            raise ValueError(f"No TypeSafe API key found. Pass api_key= or set "
                             f"the {_ENV_KEY} environment variable.")
        self.questions   = questions
        self.model       = model
        self.name        = name or type(self).name
        self.description = description or _describe(questions)
        self._api_key    = key
        self._http       = make_http(timeout=DEFAULT_TIMEOUT, retries=_RETRIES)

    def run(self, state, options=None) -> dict:
        """
        Answer the tool's questions about *state*.

        *state* is text from a model; from code it may also be a dict or a
        list — the API takes structured state, and a record keeps its fields
        that way rather than being flattened into a string.

        Returns ``{"answers": {...}, "model": <versioned id that answered>,
        "usage": {...}, "cost": <USD>}``. Raises ``RuntimeError`` on a non-2xx
        answer, with the status and the API's own message.
        """
        if state is None or (isinstance(state, str) and not state.strip()):
            raise ValueError("decideTypeSafe needs a state to judge — it was empty")
        body = {"state": state, "model": self.model, "questions": self.questions}
        response = self._http.request(
            "POST", _ENDPOINT,
            body=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self._api_key}",
                     "Content-Type": "application/json"},
        )
        raw = response.data.decode("utf-8", errors="replace")
        if not 200 <= response.status < 300:
            raise RuntimeError(f"TypeSafe API error [{response.status}]: {raw}")
        data = json.loads(raw)
        answers = data.get("answers")
        if not isinstance(answers, dict):
            raise RuntimeError(f"TypeSafe returned no answers: {raw}")
        usage = data.get("usage") or {}
        return {
            "answers": answers,
            # The alias resolves on the server; this is the version that
            # answered, the one to log beside a threshold.
            "model":   data.get("model", self.model),
            "usage":   usage,
            "cost":    (usage.get("input_tokens") or 0) * _PRICE_PER_INPUT_TOKEN,
        }
