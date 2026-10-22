"""Inception's Mercury Decide (``POST https://api.inceptionlabs.ai/v1/decisions``)
behind the common :class:`~tools.decision.Decision` format. Its wire is
TypeSafe's — ``noul`` for a predicate, ``criteria`` as a map for a choice and a
list for a score, a score's probabilities keyed by level index — so the
translation is shared; what differs is that it looks at images, which travel as
``image_url`` parts beside the text in ``state``. A dict or list state travels
as structure when there are no images. Key: ``INCEPTION_API_KEY``. See
docs/tools/decide.md and https://docs.inceptionlabs.ai/capabilities/decisions.
"""

from __future__ import annotations

import json

from ._base import Decision
from .typesafe import _answer, _wire_questions

_ENDPOINT = "https://api.inceptionlabs.ai/v1/decisions"


class decisionInception(Decision):
    """
    Typed questions about a text — and images — answered with probabilities,
    by Inception's Mercury Decide.

    ``model`` is ``"mercury-decide"``, the only one the endpoint serves. Input
    costs $0.02 per million tokens at the launch price ($0.04 regular); output
    is free. Up to 16 images a request, each a base64 data URL of at most
    10 MiB — the endpoint takes no hosted URLs.

    Examples
    --------
    ::

        from yait_aichain.tools import decisionInception

        triage = decisionInception(name="triage", questions={
            "urgent": {"type": "predicate", "instructions": "Does this convey urgency?"},
            "team":   {"type": "choice", "instructions": "Which team should handle it?",
                       "choices": {"billing": "Payments, refunds", "technical": "Bugs, outages"}},
        })
        triage.run(state="Help! My payouts have been failing for 3 days.")
    """

    name                   = "decisionInception"
    _ENV_KEY               = "INCEPTION_API_KEY"
    _PROVIDER              = "Inception"
    _DEFAULT_MODEL         = "mercury-decide"
    #: "Mercury Decide costs $0.02 per million input tokens (50% off the regular
    #: $0.04 price). Only input tokens are billed" — docs.inceptionlabs.ai,
    #: capabilities/decisions, read 2026-10-21.
    _PRICE_PER_INPUT_TOKEN = 0.02 / 1_000_000
    _PRICE_SOURCE          = "docs.inceptionlabs.ai/capabilities/decisions, read 2026-10-21"
    images                 = True
    #: "up to 16 images" a request, and the endpoint refuses more.
    MAX_IMAGES             = 16

    _ENDPOINT              = _ENDPOINT
    _API_NAME              = "Inception Decisions API"
    _answer                = staticmethod(_answer)

    def _body(self, state, images):
        if len(images) > self.MAX_IMAGES:
            raise ValueError(f"decisionInception takes at most {self.MAX_IMAGES} images a request; "
                             f"got {len(images)}")
        if images:
            # The image part must have exactly this shape: a data URL as a plain
            # string is read as text, and the model does not see the picture.
            text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
            state = [{"type": "text", "text": text},
                     *({"type": "image_url", "image_url": {"url": url}} for url in images)]
        return {"model": self.model, "state": state, "questions": _wire_questions(self.questions)}

    def _raw_answers(self, data):
        answers = data.get("answers")
        return answers if isinstance(answers, dict) else None
