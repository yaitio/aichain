"""
The library's size is a measured quantity, like its coverage.

Lines of code — non-empty, outside docstrings and comments — per subpackage,
held in `tests/fixtures/size_budget.json`: a subpackage that grows more than 5 %
past its number fails here, and raising the number is a change a reviewer
sees. Beside it, two ratchets that only go down: docstrings longer than their
kind allows, and docstrings that tell history — dates, versions, "used to" —
which belongs in CHANGELOG.md.

    python tests/test_size_budget.py --write    # record the sizes as they are now
"""

import ast
import io
import json
import re
import sys
import tokenize
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "yait_aichain"
BUDGET = ROOT / "tests" / "fixtures" / "size_budget.json"
GROWTH = 1.05

#: Lines a docstring of each kind may run to.
LIMITS = {"module": 10, "private": 3, "class": 40, "function": 15}
#: A date, a release number, "used to": what happened, not what is.
HISTORY = re.compile(r"\b20\d\d-\d\d\b|\b\d+\.\d+\.\d+\b|\bused to\b|\buntil \d")


def _files():
    return sorted(p for p in PACKAGE.rglob("*.py") if "__pycache__" not in p.parts)


def _docstrings(tree):
    """``(kind, node, first line, last line, text)`` for every docstring."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if not (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            continue
        if isinstance(node, ast.Module):
            kind = "module"
        elif node.name.startswith("_") and not node.name.startswith("__"):
            kind = "private"
        else:
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
        yield kind, node, body[0].lineno, body[0].end_lineno, body[0].value.value


def code_lines(path: Path) -> int:
    source = path.read_text()
    doc = set()
    for _, _, first, last, _ in _docstrings(ast.parse(source)):
        doc |= set(range(first, last + 1))
    comments = {t.start[0] for t in tokenize.generate_tokens(io.StringIO(source).readline)
                if t.type == tokenize.COMMENT}
    return sum(1 for i, line in enumerate(source.splitlines(), 1)
               if line.strip() and i not in doc and not (i in comments and line.strip().startswith("#")))


def _group(path: Path) -> str:
    rel = path.relative_to(PACKAGE)
    return rel.parts[0] if len(rel.parts) > 1 else "(top)"


def sizes() -> dict:
    out: dict = {}
    for path in _files():
        out[_group(path)] = out.get(_group(path), 0) + code_lines(path)
    return dict(sorted(out.items()))


def long_docstrings() -> list:
    found = []
    for path in _files():
        for kind, node, first, last, _ in _docstrings(ast.parse(path.read_text())):
            if last - first + 1 > LIMITS[kind]:
                found.append(f"{path.relative_to(ROOT)}:{first} {kind} {last - first + 1}")
    return found


def history_docstrings() -> list:
    found = []
    for path in _files():
        for _, _, first, _, text in _docstrings(ast.parse(path.read_text())):
            if HISTORY.search(text):
                found.append(f"{path.relative_to(ROOT)}:{first}")
    return found


def measure() -> dict:
    return {"code": sizes(), "long_docstrings": len(long_docstrings()),
            "history_docstrings": len(history_docstrings())}


class TestTheSizeBudget(unittest.TestCase):

    def setUp(self):
        self.budget = json.loads(BUDGET.read_text())
        self.now = measure()

    def test_no_subpackage_grows_past_its_budget(self):
        for group, lines in self.now["code"].items():
            with self.subTest(subpackage=group):
                allowed = self.budget["code"].get(group)
                self.assertIsNotNone(allowed, f"{group} has no budget; run --write")
                self.assertLessEqual(lines, int(allowed * GROWTH),
                                     f"{group}: {lines} lines of code, budget {allowed}; "
                                     "if the growth is meant, record it: "
                                     "python tests/test_size_budget.py --write")

    def test_the_docstring_ratchets_only_go_down(self):
        for key in ("long_docstrings", "history_docstrings"):
            with self.subTest(ratchet=key):
                self.assertLessEqual(self.now[key], self.budget[key])

    def test_no_ratchet_is_left_slack(self):
        for key in ("long_docstrings", "history_docstrings"):
            with self.subTest(ratchet=key):
                self.assertEqual(self.now[key], self.budget[key],
                                 f"{key} went down to {self.now[key]}: lower it in the budget "
                                 "(python tests/test_size_budget.py --write)")


if __name__ == "__main__":
    if "--write" in sys.argv:
        BUDGET.write_text(json.dumps(measure(), indent=2) + "\n")
        print(BUDGET.read_text())
    elif "--list" in sys.argv:
        print("\n".join(long_docstrings() if "long" in sys.argv else history_docstrings()))
    else:
        unittest.main()
