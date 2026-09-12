"""
One input, every text provider: does the same request mean the same thing?

Provider interchangeability is the library's main promise — "change one word
to switch providers" is on the front page — and the places it has broken have
all been the same shape: a request that works on six providers and quietly
does something else on the seventh. Google entered its image branch on
`modalities` while everyone else was asked with `format.type == "image"`, so
the same call rendered a picture six times and text once. The probe that was
supposed to catch it was not touching the path.

So this asks one question per axis, of every provider at once, and the value
is in the *comparison* rather than in any single answer. A per-provider test
written by whoever wired that provider agrees with itself by construction;
a table asks whether they agree with each other.

Nothing here talks to a network: a request is built and inspected. What a
provider *does* with what it receives is the parameter matrix's job
(`scripts/parameters.py`), established by effect against the live APIs.
"""

import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yait_aichain import Model                                  # noqa: E402
from yait_aichain.models._base import models as _models         # noqa: E402
from yait_aichain.models._data import PROVIDERS                 # noqa: E402

#: One text model per provider. The image-only ones are excluded by name
#: rather than skipped silently — a provider that grows a text model should
#: turn up here rather than stay quietly out of the comparison.
IMAGE_ONLY = {"bfl", "recraft", "reve", "private"}

SYSTEM = {"role": "system", "parts": [{"type": "text",
                                       "text": "You are terse."}]}
USER = {"role": "user", "parts": [{"type": "text", "text": "Say hi."}]}


def _text_models():
    for provider in sorted(PROVIDERS):
        if provider in IMAGE_ONLY:
            continue
        names = [n for n in _models(provider=provider)
                 if "image" not in n and "edit" not in n]
        assert names, f"{provider} has no text model to compare"
        yield provider, names[0]


def _build(name, output, messages=(SYSTEM, USER)):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Model(name, api_key="k").to_request(list(messages), output)


def _find(obj, key):
    """Every value stored under *key*, at any depth."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                found.append(v)
            found += _find(v, key)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            found += _find(v, key)
    return found


class TestTheSameInputReachesEveryProvider(unittest.TestCase):

    def test_a_system_prompt_is_never_dropped(self):
        """It may be a top-level field or a first message — the providers
        disagree and that is fine. Vanishing is not: an instruction the caller
        wrote and the model never saw changes the answer and leaves no trace."""
        for provider, name in _text_models():
            with self.subTest(provider=provider):
                _, body = _build(name, {"format": {"type": "text"}})
                self.assertIn("You are terse.", str(body),
                              f"{provider}: the system prompt did not travel")

    def test_two_system_messages_both_survive(self):
        """One system message is the common shape, and it is why this held
        for so long: the Responses path **assigned** rather than accumulated,
        so a second message replaced the first and only the last was sent.
        Every other family concatenates, so the same conversation carried
        different instructions depending on the model — silently.

        `gpt-5.5` is named explicitly because it is the model that routes
        through that path; the loop below would not otherwise reach it."""
        two = [{"role": "system", "parts": [{"type": "text", "text": "FIRST"}]},
               {"role": "system", "parts": [{"type": "text", "text": "SECOND"}]},
               USER]
        names = [n for _, n in _text_models()] + ["gpt-5.5"]
        for name in names:
            with self.subTest(model=name):
                _, body = _build(name, {"format": {"type": "text"}}, two)
                self.assertIn("FIRST", str(body),
                              f"{name}: the first system message was dropped")
                self.assertIn("SECOND", str(body))

    def test_the_user_text_arrives(self):
        for provider, name in _text_models():
            with self.subTest(provider=provider):
                _, body = _build(name, {"format": {"type": "text"}})
                self.assertIn("Say hi.", str(body))

    def test_json_mode_asks_for_json_somewhere(self):
        """A unified `format: json` is still an open item — the field is the
        provider's own — so this asserts the intent survives, not its
        spelling. The day one provider drops it entirely, this is what
        notices."""
        undelivered = []
        for provider, name in _text_models():
            _, body = _build(name, {"format": {"type": "json"}})
            if "json" not in str(body).lower():
                undelivered.append(provider)
        self.assertEqual(undelivered, [],
                         f"json was asked for and did not reach: {undelivered}")

    def test_a_ceiling_on_the_answer_always_travels(self):
        """Under whichever name — `max_tokens`, `max_completion_tokens`,
        `maxOutputTokens`. An unbounded answer is a bill."""
        for provider, name in _text_models():
            with self.subTest(provider=provider):
                _, body = _build(name, {"format": {"type": "text"}})
                names = ("max_tokens", "max_completion_tokens",
                         "maxOutputTokens")
                self.assertTrue(
                    any(_find(body, n) for n in names),
                    f"{provider}: no ceiling on the answer's length")

    def test_the_model_name_is_on_the_request(self):
        for provider, name in _text_models():
            with self.subTest(provider=provider):
                path, body = _build(name, {"format": {"type": "text"}})
                self.assertIn(name, str(body) + path)


class TestWhereTheyLegitimatelyDiffer(unittest.TestCase):
    """Not every difference is a defect, and pinning the ones that are real
    keeps the tests above honest: a check that has to be loosened until it
    passes everywhere is measuring nothing."""

    def test_the_system_prompt_lands_in_two_different_places(self):
        """Anthropic and Google carry it beside the messages; the
        OpenAI-compatible family carries it as the first message. Both are
        correct, and a caller never sees the difference — which is the
        promise."""
        beside, inside = [], []
        for provider, name in _text_models():
            _, body = _build(name, {"format": {"type": "text"}})
            top = {"system", "system_instruction", "instructions"}
            (beside if top & set(body) else inside).append(provider)
        self.assertTrue(beside, "no provider carries it beside the messages")
        self.assertTrue(inside, "no provider carries it as a message")

    def test_the_ceiling_has_three_names(self):
        spellings = set()
        for provider, name in _text_models():
            _, body = _build(name, {"format": {"type": "text"}})
            for n in ("max_tokens", "max_completion_tokens",
                      "maxOutputTokens"):
                if _find(body, n):
                    spellings.add(n)
        self.assertGreater(len(spellings), 1,
                           "the wire names converged — the test above can be "
                           "tightened to one name")



# ── Shapes the examples do not contain ──────────────────────────────────────
#
# Both defects this file has found so far lived in a shape our own corpus
# cannot produce. Of 24 examples exactly one carries a system prompt — a
# single one, on Anthropic — and exactly one runs a Responses model, through
# an Agent, which emits one system message always. So the examples doubled as
# the test set and the test set never contained the case: `instructions` was
# **assigned** rather than accumulated on the Responses path since the very
# first commit, and a second system message replaced the first for three
# months.
#
# These are therefore the conversations nobody writes in a README. What is
# compared is the **meaning** — is the system text global, and what is the
# sequence of conversational turns — rather than the wire, because the wire
# legitimately differs: Google says `model` where everyone else says
# `assistant`, and three providers carry the system prompt in a field of
# their own.

def _t(role, *texts):
    return {"role": role,
            "parts": [{"type": "text", "text": t} for t in texts]}


SHAPES = {
    "a system message in the middle":
        [_t("user", "hi"), _t("system", "MID"), _t("user", "again")],
    "system messages at both ends":
        [_t("system", "A"), _t("user", "hi"), _t("system", "B")],
    "an assistant turn in the middle":
        [_t("user", "U1"), _t("assistant", "A1"), _t("user", "U2")],
    "an assistant turn first":
        [_t("assistant", "A1"), _t("user", "U1")],
    "two user turns in a row":
        [_t("user", "U1"), _t("user", "U2")],
    "a multi-part message":
        [_t("user", "PART1", "PART2")],
    "an empty part beside a real one":
        [_t("user", ""), _t("user", "REAL")],
    "a message with no parts at all":
        [{"role": "user", "parts": []}, _t("user", "REAL")],
}

#: Google's word for the assistant. A wire spelling, not a meaning.
CANONICAL_ROLE = {"model": "assistant"}


def _meaning(body):
    """(is the system prompt global?, the conversation's roles).

    Everything a caller can observe, and nothing a provider is entitled to
    spell its own way.
    """
    has_system = any(k in body for k in
                     ("system", "system_instruction", "instructions"))
    turns = []
    for key in ("messages", "contents", "input"):
        if isinstance(body.get(key), list):
            for message in body[key]:
                if not isinstance(message, dict):
                    continue
                role = message.get("role") or message.get("author") or "?"
                role = CANONICAL_ROLE.get(role, role)
                if role == "system":
                    has_system = True
                    continue
                turns.append(role)
            break
    return has_system, tuple(turns)


class TestEveryProviderIsSentTheSameConversation(unittest.TestCase):

    def test_the_shapes_no_example_contains(self):
        names = [n for _, n in _text_models()] + ["gpt-5.5"]
        for shape, messages in SHAPES.items():
            with self.subTest(shape=shape):
                readings = {}
                for name in names:
                    _, body = _build(name, {"format": {"type": "text"}},
                                     messages)
                    readings.setdefault(_meaning(body), []).append(name)
                self.assertEqual(
                    len(readings), 1,
                    f"{shape}: providers disagree — "
                    + "; ".join(f"{k} ← {v}" for k, v in readings.items()))

    def test_no_text_is_dropped_by_any_of_them(self):
        names = [n for _, n in _text_models()] + ["gpt-5.5"]
        for shape, messages in SHAPES.items():
            wanted = [p["text"] for m in messages for p in m["parts"]
                      if p["text"]]
            for name in names:
                with self.subTest(shape=shape, model=name):
                    _, body = _build(name, {"format": {"type": "text"}},
                                     messages)
                    for text in wanted:
                        self.assertIn(text, str(body))


class TestASystemMessageIsGlobalWhereverItSits(unittest.TestCase):
    """The meaning had to be *chosen*, and only one choice was available.
    Anthropic, Google and the Responses API each carry the system prompt in a
    field of their own, so a system message written mid-conversation is
    hoisted out of the sequence and applies from the start. The
    OpenAI-compatible family left it in place, where it reads as an
    instruction beginning at that point — the same conversation meaning two
    things. Three providers cannot express the positional reading at all, so
    global is the only portable one."""

    def test_it_is_hoisted_on_the_openai_family_too(self):
        _, body = _build("gpt-4o", {"format": {"type": "text"}},
                         SHAPES["a system message in the middle"])
        roles = [m["role"] for m in body["messages"]]
        self.assertEqual(roles[0], "system",
                         "a mid-conversation system message stayed in place")
        self.assertNotIn("system", roles[1:])

    def test_and_the_caller_is_told(self):
        """It is a real change to what that family's models see, so it is an
        adaptation and not a silent tidy-up."""
        model = Model("gpt-4o", api_key="k")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.to_request(SHAPES["a system message in the middle"],
                             {"format": {"type": "text"}})
        self.assertIn("system message position",
                      {a.option for a in model.last_adaptations})

    def test_a_conversation_with_no_system_message_is_untouched(self):
        model = Model("gpt-4o", api_key="k")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.to_request(SHAPES["two user turns in a row"],
                             {"format": {"type": "text"}})
        self.assertEqual(model.last_adaptations, [])


if __name__ == "__main__":
    unittest.main()
