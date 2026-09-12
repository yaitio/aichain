"""
The promise on the front page has to still be true.

`VISION.md` states it as an invariant rather than a wish: *the "hello world"
from the README works without a single new MANDATORY parameter, and every new
capability enters as an option with a default under which local mode is zero
configuration.* Surface area grows; the entry barrier does not move. And it is
called an architectural guarantee "because all four mechanisms are about the
environment, and the environment is injected from outside the scenario — so if
a simple `Skill.run()` got more complex, that is a signal the environment
leaked into the scenario."

That signal had nothing watching for it. `PLAN.md` has listed this test as
missing since the June audit, and in the meantime the README's *agent* example
acquired a `TypeError` on its first line and kept it through four minor
versions. This is the check, in two halves:

* the hello world **runs**, against a fake transport, with exactly the
  arguments the page shows;
* every other fenced example **binds** to the real signatures — required
  arguments supplied, no keyword that does not exist.

The second half is static on purpose. Executing every example would need
credentials, a network and a tolerance for flakes; binding needs none of
those and catches the whole class the agent example belonged to.
"""

import ast
import inspect
import json
import re
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from yait_aichain import Agent, Chain, Model, Pool, Skill

ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted((ROOT / "docs").rglob("*.md")) + [ROOT / "README.md"]

#: Historical records: they describe a past version on purpose.
EXEMPT = ("docs/design",)

CLASSES = {"Agent": Agent, "Chain": Chain, "Skill": Skill, "Pool": Pool}


def _blocks(text: str):
    return re.findall(r"```python\n(.*?)```", text, re.DOTALL)


def _pages():
    for page in PAGES:
        rel = page.relative_to(ROOT).as_posix()
        if not rel.startswith(EXEMPT):
            yield rel, page.read_text()


class TestTheHelloWorldStillRuns(unittest.TestCase):
    """Exactly the arguments the front page shows — no more."""

    def _answering(self, model):
        reply = json.dumps({
            "content": [{"type": "text", "text": "A summary."}],
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }).encode()
        model.client._post = MagicMock(return_value=reply)
        model.client._auth_headers = MagicMock(return_value={})
        return model

    def test_two_arguments_and_a_run(self):
        skill = Skill(
            model=self._answering(Model("claude-sonnet-4-6", api_key="k")),
            input={"messages": [{"role": "user",
                                 "parts": ["Summarise: {text}"]}]},
        )
        self.assertEqual(skill.run(variables={"text": "..."}), "A summary.")

    def test_the_provider_swaps_with_one_word(self):
        """The sentence under the example — "change this one word" — is the
        library's whole pitch, so it is a test and not a claim."""
        for name in ("claude-sonnet-4-6", "gpt-4o", "gemini-2.5-pro",
                     "grok-3"):
            with self.subTest(model=name):
                skill = Skill(
                    model=Model(name, api_key="k"),
                    input={"messages": [{"role": "user",
                                         "parts": ["Summarise: {text}"]}]},
                )
                path, body = skill.models[0].to_request(
                    [{"role": "user", "parts": [{"type": "text",
                                                 "text": "hi"}]}],
                    {"format": {"type": "text"}})
                self.assertTrue(path and body)

    def test_the_readme_still_shows_that_example(self):
        """The test is only an invariant while it is testing the page. If the
        front page changes its opening example, this has to be looked at
        rather than silently passing on an example nobody reads."""
        first = _blocks((ROOT / "README.md").read_text())[0]
        self.assertIn("Skill(", first)
        self.assertIn("skill.run(", first)
        tree = ast.parse(first)
        call = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.Call)
                    and getattr(n.func, "id", None) == "Skill")
        self.assertEqual({kw.arg for kw in call.keywords}, {"model", "input"})


class TestEveryExampleBindsToTheRealThing(unittest.TestCase):
    """`test_docs_promise_what_exists` checks that a named keyword exists.
    This checks the other direction — that the call could actually be made:
    required arguments supplied, arity right. The agent examples had both
    problems, and only the first was being caught."""

    def test_every_call_binds(self):
        failures = []
        for rel, text in _pages():
            for block in _blocks(text):
                try:
                    tree = ast.parse(block)
                except SyntaxError:
                    continue                    # a fragment, not an example
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    name = (getattr(node.func, "id", None)
                            or getattr(node.func, "attr", None))
                    cls = CLASSES.get(name)
                    if cls is None:
                        continue
                    # Values are irrelevant: this asks whether the call shape
                    # is one the constructor accepts, not whether it works.
                    args = [None] * len(node.args)
                    if any(isinstance(a, ast.Starred) for a in node.args):
                        continue
                    kwargs = {kw.arg: None for kw in node.keywords if kw.arg}
                    try:
                        inspect.signature(cls.__init__).bind(None, *args,
                                                             **kwargs)
                    except TypeError as exc:
                        failures.append(f"{rel}: {name}(...) — {exc}")
        self.assertEqual(sorted(set(failures)), [],
                         "\n".join(sorted(set(failures))))


if __name__ == "__main__":
    unittest.main()
