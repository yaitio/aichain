"""
tools.decision._base
====================

``Decision`` — typed questions about a text, answered with probabilities.

A decision model writes no prose. It is given evidence and a fixed set of
questions, and answers each one with a number: the probability that a
statement is true, one option from a fixed set with a probability for every
option, or a place on an ordered scale. Two providers sell exactly this —
TypeSafe's Jev and OpenAI's Decisions API — and their wire formats differ in
every detail while their meaning does not. This class holds the meaning; each
provider translates it.

Questions — one format for every provider
-----------------------------------------
A map of name → question. The name is what the answer comes back under.

``predicate``
    ``{"type": "predicate", "instructions": "Is this urgent?"}``
``choice``
    ``{"type": "choice", "instructions": "Which team?",
    "choices": {"billing": "Payments, refunds", "other": None}}`` — option →
    description, ``None`` when an option needs none; 2 to 255 options.
``score``
    ``{"type": "score", "instructions": "How severe?",
    "levels": ["Cosmetic", {"label": "Blocked", "description": "No workaround"}]}``
    — ordered from lowest to highest, 2 to 10 levels, each a label or a
    label with a description.

Answers — one format for every provider
---------------------------------------
``run()`` returns ``{"answers", "model", "usage", "cost", "cost_basis"}`` as a
:class:`~tools.Record` — the fields for code, one line per answer for a model
(:meth:`Decision.render`); ``answers`` maps each question's name to::

    {"type": "predicate", "probability": 0.92}
    {"type": "choice", "choice": "billing", "confidence": 0.93,
     "probabilities": {"billing": 0.95, "other": 0.05}}
    {"type": "score", "score": 1.1, "confidence": 0.55,
     "probabilities": {"Cosmetic": 0.1, "Workaround": 0.7, "Blocked": 0.2}}

A score is the probability-weighted average of the level indices, from 0, so
it can fall between levels. Probabilities are returned as the provider gave
them; the threshold is the caller's, who knows what a wrong answer costs.

The questions are fixed when the tool is made, and an agent passes only the
text, as ``state``. A model that wrote the questions could write them badly,
and the calibration that is the reason to use one of these would go with them.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os

from ...clients._base import make_http
from ...clients._constants import DEFAULT_RETRIES, DEFAULT_TIMEOUT
from .._base import Record, Tool

QUESTION_TYPES = ("predicate", "choice", "score")
#: Both providers' own ceilings: 255 options, 2–10 levels. OpenAI's checked
#: live on 2026-10-08 (256 and 11 are refused with HTTP 400), TypeSafe's read
#: from its documentation on 2026-10-02.
MAX_CHOICES = 255
SCORE_LEVELS = (2, 10)


def _level(level) -> tuple[str, "str | None"]:
    """A level as (label, description)."""
    if isinstance(level, str):
        return level, None
    return str(level.get("label") or ""), level.get("description")


def check_questions(questions, owner: str = "Decision") -> None:
    """Refuse a malformed question here, naming it, rather than as a 400."""
    if not isinstance(questions, dict) or not questions:
        raise ValueError(
            f"{owner} needs at least one question, as a map of name → question, "
            "e.g. {'urgent': {'type': 'predicate', 'instructions': 'Is this urgent?'}}")
    for name, q in questions.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"every question needs a name; got {name!r}")
        if not isinstance(q, dict):
            raise ValueError(f"question {name!r} must be a mapping, got {type(q).__name__}")
        kind = q.get("type")
        if kind not in QUESTION_TYPES:
            hint = "; 'noul' is TypeSafe's word for predicate" if kind == "noul" else ""
            raise ValueError(f"question {name!r} has type {kind!r}; it must be one of "
                             f"{', '.join(QUESTION_TYPES)}{hint}")
        if not isinstance(q.get("instructions"), str) or not q["instructions"].strip():
            raise ValueError(f"question {name!r} has no instructions — that is the question itself")
        unknown = set(q) - {"type", "instructions", "choices", "levels"}
        if unknown:
            raise ValueError(f"question {name!r} has unknown keys {sorted(unknown)}; a question "
                             "takes type, instructions and, by type, choices or levels")
        if kind == "choice":
            ch = q.get("choices")
            if not isinstance(ch, dict) or len(ch) < 2:
                raise ValueError(f"choice {name!r} needs choices: a map of option → "
                                 "description (None when an option needs none), at least two")
            if len(ch) > MAX_CHOICES:
                raise ValueError(f"choice {name!r} has {len(ch)} options; at most {MAX_CHOICES}")
        elif "choices" in q:
            raise ValueError(f"{kind} {name!r} takes no choices")
        if kind == "score":
            lv = q.get("levels")
            lo, hi = SCORE_LEVELS
            if not isinstance(lv, list) or not lo <= len(lv) <= hi:
                raise ValueError(f"score {name!r} needs levels: an ordered list of {lo} to {hi}, "
                                 "lowest first, each a label or {'label', 'description'}")
            labels = [_level(x)[0] for x in lv]
            if not all(labels) or len(set(labels)) != len(labels):
                raise ValueError(f"score {name!r}: every level needs a label, and labels "
                                 "must differ — the answer is keyed by them")
        elif "levels" in q:
            raise ValueError(f"{kind} {name!r} takes no levels")


def describe(questions: dict) -> str:
    """What the tool tells the model it will learn — one line per question."""
    lines = []
    for name, q in questions.items():
        if q["type"] == "predicate":
            what = "probability that the answer is yes"
        elif q["type"] == "choice":
            what = "one of " + ", ".join(map(str, q["choices"])) + ", with probabilities"
        else:
            labels = [_level(x)[0] for x in q["levels"]]
            what = f"a score from 0 to {len(labels) - 1} on: " + " < ".join(labels)
        lines.append(f"- {name}: {q['instructions']} → {what}")
    return ("Judge a text against fixed questions and get calibrated answers back — "
            "probabilities, not prose. Pass the text as `state`.\n" + "\n".join(lines))


def image_url(image) -> str:
    """An image as the base64 data URL a decision endpoint takes — from a data
    URL, raw bytes, or a local path."""
    if isinstance(image, str) and image.startswith("data:"):
        return image
    if isinstance(image, (bytes, bytearray)):
        data, mime = bytes(image), "image/png"
    elif isinstance(image, str) and os.path.isfile(image):
        with open(image, "rb") as fh:
            data = fh.read()
        mime = mimetypes.guess_type(image)[0] or "image/png"
    else:
        raise ValueError("an image must be a data: URL, bytes or a local file path; "
                         "hosted URLs are not taken by the decision endpoints")
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def _p(v) -> str:
    return "?" if v is None else f"{v:.2f}"


class Decision(Tool):
    """
    Base of the decision tools: one format of questions and answers, a
    provider per subclass.

    Parameters (every provider)
    ---------------------------
    questions : dict
        Map of name → question, in the format above. Fixed for the life of
        the tool; checked here, before any network.
    name : str, optional
        The name the model sees. Name it by what it decides — ``"triage"``,
        ``"is_duplicate"`` — and an agent sees ``triage(state)``.
    description : str, optional
        Default: written from the questions, so the model knows what it will
        learn by calling it.
    model : str, optional
        The provider's model; each subclass has its default.
    api_key : str, optional
        Overrides the provider's environment variable.

    A subclass sets ``name``, ``_ENV_KEY``, ``_PROVIDER``,
    ``_PRICE_PER_INPUT_TOKEN``, ``_DEFAULT_MODEL``, ``images`` (whether it takes
    them) and, if its API needs other retries, ``_RETRIES``; and says how its
    API is asked and answers — see ``_ask``.
    """

    name        = "decision"
    description = "Judge a text against fixed typed questions; returns probabilities."
    # It reads the evidence it is given and changes nothing anywhere.
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

    #: The provider's key variable; a tuple here so the base names none.
    _ENV_KEY: "str | tuple" = ()
    _PROVIDER: str = ""
    _RETRIES = DEFAULT_RETRIES
    _DEFAULT_MODEL: str = ""
    _PRICE_PER_INPUT_TOKEN: float = 0.0
    #: Where that rate was read, and when.
    _PRICE_SOURCE: str = ""
    #: Whether the provider takes images beside the text.
    images: bool = False

    def __init__(self, questions: dict, *, name: "str | None" = None,
                 description: "str | None" = None, model: "str | None" = None,
                 api_key: "str | None" = None) -> None:
        check_questions(questions, type(self).__name__)
        key = api_key or os.environ.get(self._ENV_KEY)
        if not key:
            raise ValueError(f"No {self._PROVIDER} API key found. Pass api_key= or set the "
                             f"{self._ENV_KEY} environment variable.")
        self.questions   = questions
        self.model       = model or self._DEFAULT_MODEL
        self.name        = name or type(self).name
        self.description = description or describe(questions)
        self._api_key    = key
        self._http       = make_http(timeout=DEFAULT_TIMEOUT, retries=self._RETRIES)

    def run(self, state, options=None, *, images=None) -> dict:
        """
        Answer the tool's questions about *state*.

        *state* is text from a model; from code it may also be a dict or a
        list, which a provider that takes structure receives as structure and
        one that does not receives as JSON text. *images* — data URLs, bytes or
        local paths — go to a provider that takes them and are refused, by
        name, by one that does not.

        Returns ``{"answers", "model", "usage", "cost"}`` — answers in the
        common format, the model version that answered, input and output
        tokens, and the cost in USD (input only; both providers bill nothing
        else). Raises ``RuntimeError`` on a non-2xx answer, with the status
        and the provider's own message.
        """
        if state is None or (isinstance(state, str) and not state.strip()):
            raise ValueError(f"{type(self).__name__} needs a state to judge — it was empty")
        if images and not self.images:
            raise ValueError(f"{type(self).__name__} ({self._PROVIDER}) takes no images; "
                             "decisionOpenAI does")
        answers, model, usage = self._ask(state, [image_url(i) for i in images or []])
        return Record({"answers": answers, "model": model, "usage": usage,
                       "cost": (usage.get("input_tokens") or 0) * self._PRICE_PER_INPUT_TOKEN,
                       "cost_basis": f"input tokens reported by {self._PROVIDER}, at the published "
                                     f"rate ({self._PRICE_SOURCE})"},
                      self.render)

    def render(self, out: dict) -> str:
        """What a model reads: one line per question, with the probabilities."""
        lines = []
        for name, a in out["answers"].items():
            if a["type"] == "predicate":
                lines.append(f"{name}: yes with probability {_p(a['probability'])}")
            elif a["type"] == "choice":
                rest = ", ".join(f"{k} {_p(v)}" for k, v in (a.get("probabilities") or {}).items())
                lines.append(f"{name}: {a.get('choice')} (confidence {_p(a.get('confidence'))}; {rest})")
            else:
                rest = ", ".join(f"{k} {_p(v)}" for k, v in (a.get("probabilities") or {}).items())
                lines.append(f"{name}: score {_p(a.get('score'))} (confidence {_p(a.get('confidence'))}; {rest})")
        return "\n".join(lines)

    def _ask(self, state, images: list) -> tuple[dict, str, dict]:
        """One request: the answers by question, the model that answered, and
        usage. A provider gives `_ENDPOINT`, `_API_NAME`, `_body(state, images)`,
        `_raw_answers(data)` — by question name, or None — and `_answer(q, a)`."""
        response = self._http.request(
            "POST", self._ENDPOINT, body=json.dumps(self._body(state, images)).encode("utf-8"),
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"})
        raw = response.data.decode("utf-8", errors="replace")
        if not 200 <= response.status < 300:
            raise RuntimeError(f"{self._API_NAME} error [{response.status}]: {raw}")
        data = json.loads(raw)
        answers = self._raw_answers(data)
        if answers is None:
            raise RuntimeError(f"{self._PROVIDER} returned no answers: {raw}")
        usage = data.get("usage") or {}
        return ({n: self._answer(q, answers.get(n) or {}) for n, q in self.questions.items()},
                data.get("model", self.model),
                {"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")})

    def _serialise_init_args(self) -> dict:
        """What `Chain.save` writes: the questions and the model, never the key."""
        out = {"questions": self.questions, "model": self.model}
        if self.name != type(self).name:
            out["name"] = self.name
        if self.description != describe(self.questions):
            out["description"] = self.description
        return out

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, model={self.model!r}, questions={list(self.questions)})"
