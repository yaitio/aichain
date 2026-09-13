"""
Summarise a live-test run for the job page, and fail a run that measured nothing.

    python scripts/live_summary.py live.xml >> "$GITHUB_STEP_SUMMARY"

The live tests skip a provider whose key is absent, which is right for a
developer's machine and wrong for a scheduled job: a run where every class
skipped is green and says nothing. So the table counts passed, failed and
skipped per class, and the exit status is 1 when nothing passed.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from collections import defaultdict


def tally(path: str) -> dict:
    counts: dict = defaultdict(lambda: {"passed": 0, "failed": 0, "skipped": 0})
    for case in ET.parse(path).getroot().iter("testcase"):
        cls = case.get("classname", "?").rsplit(".", 1)[-1]
        if case.find("failure") is not None or case.find("error") is not None:
            counts[cls]["failed"] += 1
        elif case.find("skipped") is not None:
            counts[cls]["skipped"] += 1
        else:
            counts[cls]["passed"] += 1
    return dict(counts)


def render(counts: dict) -> str:
    rows = ["| Class | Passed | Failed | Skipped |", "|---|---|---|---|"]
    for cls in sorted(counts):
        c = counts[cls]
        rows.append(f"| `{cls}` | {c['passed']} | {c['failed']} | {c['skipped']} |")
    total = {k: sum(c[k] for c in counts.values()) for k in ("passed", "failed", "skipped")}
    rows.append(f"| **total** | **{total['passed']}** | **{total['failed']}** | **{total['skipped']}** |")
    head = "## Live tests\n\n"
    if total["passed"] == 0:
        head += ("**Nothing passed.** Every class skipped or failed — check that the "
                 "provider keys are set as repository secrets.\n\n")
    return head + "\n".join(rows) + "\n"


def main(argv) -> int:
    try:
        counts = tally(argv[0])
    except (IndexError, FileNotFoundError, ET.ParseError) as exc:
        print(f"## Live tests\n\nNo results to summarise: {exc}")
        return 1
    print(render(counts))
    return 0 if sum(c["passed"] for c in counts.values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
