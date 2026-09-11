"""
The docs must not name an API that is gone.

Three whole flows were documented for `Agent` and could not run: a `store=`
parameter, `agent.resume()`, `agent.context`, `SuspendedResult` from
`agent.run()`, and `allow_spawn=`/`spawn_agent` under their pre-rename names.
`2.0.0` removed the agent's suspend/resume deliberately — its state *is* the
conversation — and the pages that taught it were never touched. One of them
was the cross-process serverless pattern, which is the flagship story.

This is the cheap half of "snippets are extracted, never typed": it does not
run the examples, it only checks that the names they use exist. A test that
fails when the code drops something the docs still teach is worth more than
the six lines it costs, because the alternative is a reader finding out.
"""

import ast
import re
import unittest
from pathlib import Path

from yait_aichain import Agent, Chain, Skill

ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted((ROOT / "docs").rglob("*.md")) + [ROOT / "README.md"]

#: Historical records, not instructions: they describe what was planned at a
#: past version and are wrong on purpose the moment the plan changed.
EXEMPT = {"docs/design"}

#: Read with `ast`, not with a regex. `Skill(model=Model("gpt-4o",
#: api_key="k"))` stops a non-greedy `[^)]*` at the inner bracket, so the
#: inner call's keywords are read as the outer one's and every nested example
#: reports a parameter that is not missing. A parser knows where a call ends.

#: How many documented parameters do not exist. **34** the day this test was
#: written, and every one of them is the agent: the 2.0 rewrite renamed
#: `orchestrator` to `model` and folded `max_steps` / `max_tokens` /
#: `max_attempts` / `done_when` into `stop_when`, dropped `memory`, `store`
#: and `executors`, and turned `allow_spawn` into `team`. All of it is written
#: down in `docs/design/default-agent.md`, decision by decision — the design
#: record was kept and the pages that teach the API were not touched, so the
#: agent documentation describes a library that has not existed since 2.0.0.
#: A reader copying the README's agent example gets a TypeError on line one.
#:
#: Lower it as pages are fixed. Never raise it.
STALE_CEILING = 34

CLASSES = {"Agent": Agent, "Chain": Chain, "Skill": Skill}


def _code_blocks(text: str):
    """Fenced python blocks only. Prose may name a removed API in order to say
    it was removed — this page does exactly that."""
    return re.findall(r"```python\n(.*?)```", text, re.DOTALL)


class TestEveryDocumentedParameterExists(unittest.TestCase):

    def test_no_page_teaches_a_parameter_that_is_gone(self):
        import inspect
        signatures = {name: set(inspect.signature(cls.__init__).parameters)
                      for name, cls in CLASSES.items()}
        missing = []
        for page in PAGES:
            rel = page.relative_to(ROOT).as_posix()
            if any(rel.startswith(x) for x in EXEMPT):
                continue
            for block in _code_blocks(page.read_text()):
                try:
                    tree = ast.parse(block)
                except SyntaxError:
                    continue          # a fragment, not an example; skip it
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    fn = node.func
                    name = getattr(fn, "id", None) or getattr(fn, "attr", None)
                    if name not in signatures:
                        continue
                    for kw in node.keywords:
                        if kw.arg and kw.arg not in signatures[name]:
                            missing.append(f"{rel}: {name}({kw.arg}=...)")
        found = sorted(set(missing))
        self.assertLessEqual(
            len(found), STALE_CEILING,
            f"{len(found)} documented parameters do not exist, ceiling is "
            f"{STALE_CEILING}:\n" + "\n".join(found))
        self.assertEqual(
            len(found), STALE_CEILING,
            f"{len(found)} left but the ceiling is {STALE_CEILING} — lower it "
            "to match, or the ratchet stops holding anything.")


class TestTheRemovedAgentStateApiStaysRemoved(unittest.TestCase):
    """Named individually because each was a worked example a reader would
    have copied, and because the reason they are gone is a design decision
    worth not re-losing: the agent's state is the conversation."""

    GONE = ("agent.resume(", "agent.context", "Agent(store=")

    def test_no_fenced_example_uses_it(self):
        offenders = []
        for page in PAGES:
            rel = page.relative_to(ROOT).as_posix()
            if any(rel.startswith(x) for x in EXEMPT):
                continue
            for block in _code_blocks(page.read_text()):
                for name in self.GONE:
                    if name in block:
                        offenders.append(f"{rel}: {name}")
        self.assertEqual(sorted(set(offenders)), [],
                         "\n".join(sorted(set(offenders))))

    def test_and_the_agent_really_does_not_have_it(self):
        """The other direction: if suspend/resume ever comes back to the
        agent, this test is the place the decision gets reopened out loud."""
        self.assertFalse(hasattr(Agent, "resume"))
        self.assertFalse(hasattr(Agent, "context"))


if __name__ == "__main__":
    unittest.main()
