"""
A constructor parameter that is stored is read somewhere.

`Agent(description=...)` was documented as what a lead reads when it picks a
worker, stored as `self.description`, and read by nothing — for two months,
through a full audit. Reading the code file by file cannot find this: there is
no wrong line, only a missing one. It takes a cross-reference — where an
attribute is written against where it is read — so it is a check, not a review.

The rule: a parameter of `__init__` assigned to `self.X` must be read as
`self.X` somewhere in the class's family (its bases and subclasses), or as
`.X` / `getattr(_, "X")` / a field-name list entry in a module that defines a
family member. A dict key does not count — every JSON schema has a
"description" key, and counting those hid this very defect.

What it leaves out on purpose is a public **output**: an exception's fields, a
result's fields — data handed to the caller, which the library has no reason
to read back. Each is listed below with its reason; a new entry needs one too.
"""

import ast
import textwrap
import unittest
from collections import defaultdict
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "yait_aichain"

_OUT = "an exception's field, read by whoever catches it"
_RESULT = "a result's field, handed to the caller"

#: (class, attribute) → why nothing in the library reads it.
KEPT = {
    **{(cls, "_own_budget"): "read by _budget.start_run, through getattr on the owner"
       for cls in ("Agent", "Chain", "Pool")},
    ("APIError", "status"): _OUT,
    ("APIError", "message"): _OUT,
    ("APIError", "retry_after"): _OUT,
    ("TruncatedResponseError", "finish_reason"): _OUT,
    ("TruncatedResponseError", "output_tokens"): _OUT,
    ("InvalidStructuredOutputError", "problems"): _OUT,
    ("InvalidStructuredOutputError", "value"): _OUT,
    ("BudgetExceeded", "spent"): _OUT,
    ("BudgetExceeded", "what"): _OUT,
    ("UnsupportedOption", "adaptations"): _OUT,
    ("Suspend", "resume_with"): "raised by a tool; the agent and Chain read it off the exception",
    ("Suspend", "hint"): "raised by a tool; the agent and Chain read it off the exception",
    ("PoolResult", "beacons"): _RESULT,
    ("Pool", "description"): "a label for the caller's code, documented as changing nothing — "
                             "a pool is neither offered to a model nor saved",
}


def _bases(node):
    return [b.id if isinstance(b, ast.Name) else b.attr if isinstance(b, ast.Attribute) else None
            for b in node.bases]


def _reads(node) -> set:
    got = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load):
            got.add(n.attr)
        elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
              and n.func.id in ("getattr", "hasattr") and len(n.args) >= 2
              and isinstance(n.args[1], ast.Constant) and isinstance(n.args[1].value, str)):
            got.add(n.args[1].value)
        elif isinstance(n, (ast.Tuple, ast.List, ast.Set)) and n.elts and all(
                isinstance(e, ast.Constant) and isinstance(e.value, str) for e in n.elts):
            got |= {e.value for e in n.elts}
    return got


def dropped(sources: dict) -> list:
    """`(path, line, class, attribute)` for every stored parameter nothing reads.
    *sources* maps a path to its source text."""
    trees = {p: ast.parse(src) for p, src in sources.items()}
    classes = defaultdict(list)
    for p, tree in trees.items():
        for n in ast.walk(tree):
            if isinstance(n, ast.ClassDef):
                classes[n.name].append((p, n))
    children = defaultdict(set)
    for name, defs in classes.items():
        for _, n in defs:
            for b in _bases(n):
                if b:
                    children[b].add(name)

    def family(name):
        out, todo = set(), [name]
        while todo:
            c = todo.pop()
            if c in out:
                continue
            out.add(c)
            todo += list(children.get(c, ()))
            todo += [b for _, n in classes.get(c, []) for b in _bases(n) if b in classes]
        return out

    module_reads = {p: _reads(t) for p, t in trees.items()}
    found = []
    for name, defs in classes.items():
        for path, cls in defs:
            init = next((f for f in cls.body
                         if isinstance(f, ast.FunctionDef) and f.name == "__init__"), None)
            if init is None:
                continue
            params = {a.arg for a in init.args.args + init.args.kwonlyargs} - {"self"}
            stored = {}
            for n in ast.walk(init):
                if isinstance(n, ast.Assign):
                    for t in n.targets:
                        if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                                and t.value.id == "self"
                                and {x.id for x in ast.walk(n.value) if isinstance(x, ast.Name)} & params):
                            stored[t.attr] = n.lineno
            if not stored:
                continue
            fam = family(name)
            read = set().union(*(_reads(n) for c in fam for _, n in classes.get(c, [])))
            for p in {path} | {p2 for c in fam for p2, _ in classes.get(c, [])}:
                read |= module_reads[p]
            found += [(path, line, name, attr) for attr, line in stored.items() if attr not in read]
    return sorted(found)


class TestNothingStoredIsIgnored(unittest.TestCase):

    def test_every_stored_parameter_is_read_or_listed_with_a_reason(self):
        sources = {str(p.relative_to(PACKAGE.parent)): p.read_text()
                   for p in PACKAGE.rglob("*.py") if "__pycache__" not in p.parts}
        unread = [f"{path}:{line} {cls}.{attr}" for path, line, cls, attr in dropped(sources)
                  if (cls, attr) not in KEPT]
        self.assertEqual(unread, [], "stored and never read — wire it, or list it in KEPT with "
                                     "the reason nothing needs to read it")

    def test_no_reason_outlives_its_attribute(self):
        sources = {str(p.relative_to(PACKAGE.parent)): p.read_text()
                   for p in PACKAGE.rglob("*.py") if "__pycache__" not in p.parts}
        live = {(cls, attr) for _, _, cls, attr in dropped(sources)}
        self.assertEqual(sorted(set(KEPT) - live), [],
                         "these are read now, or gone; drop them from KEPT")


class TestTheCheckBites(unittest.TestCase):
    """The scanner finds the defect it was written for, and not its fix."""

    BROKEN = textwrap.dedent('''
        def _delegate_schema(team):
            names = [w.name for w in team]
            return {"worker": {"enum": names, "description": "which worker takes it"}}

        class Agent:
            def __init__(self, model, name=None, description=None):
                self.model = model
                self.name = name
                self.description = description

            def run(self):
                return self.model, _delegate_schema([self])
    ''')

    def test_a_description_stored_and_never_read_is_found(self):
        found = dropped({"agent.py": self.BROKEN})
        self.assertEqual([(c, a) for _, _, c, a in found], [("Agent", "description")])

    def test_reading_it_clears_it(self):
        fixed = self.BROKEN.replace('"description": "which worker takes it"',
                                    '"description": "; ".join(w.description for w in team)')
        self.assertEqual(dropped({"agent.py": fixed}), [])


if __name__ == "__main__":
    unittest.main()
