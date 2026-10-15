"""
What a model is shown for every tool the library ships — its name, its
description and its parameters — is held to a snapshot.

A tool's description and its parameters' descriptions are a prompt: shortening
a docstring or rewording a field changes what a model does. A change here is
made on purpose and seen in review, never as a side effect of tidying text.

    python tests/test_tool_schemas_do_not_drift.py --write
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "tests" / "fixtures" / "tool_schemas.json"
sys.path.insert(0, str(ROOT / ".github" / "scripts"))
sys.path.insert(0, str(ROOT))


def shown() -> dict:
    import docs
    out = {}
    for cls in docs.tool_classes():
        try:
            inst = docs.instance(cls)
        except ValueError:                 # a family's base, built only through its providers
            continue
        if inst is None:
            continue
        out[f"{cls.__module__}.{cls.__qualname__}"] = {
            "name": inst.name, "description": inst.description, "parameters": inst.parameters}
    return dict(sorted(out.items()))


class TestWhatAModelIsShown(unittest.TestCase):

    def test_every_tool_matches_its_snapshot(self):
        then = json.loads(SNAPSHOT.read_text())
        now = json.loads(json.dumps(shown(), default=str))
        for key in sorted(set(then) | set(now)):
            with self.subTest(tool=key):
                self.assertEqual(now.get(key), then.get(key),
                                 "what a model is shown changed; if it is meant, "
                                 "python tests/test_tool_schemas_do_not_drift.py --write")


if __name__ == "__main__":
    if "--write" in sys.argv:
        SNAPSHOT.write_text(json.dumps(shown(), indent=1, default=str, ensure_ascii=False) + "\n")
        print(f"{len(json.loads(SNAPSHOT.read_text()))} tools")
    else:
        unittest.main()
