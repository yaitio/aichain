"""
Hold the coverage number, and make every large gap carry a reason.

    pytest -m "not live" --cov=yait_aichain --cov-report=json:coverage.json
    python .github/scripts/coverage_register.py coverage.json

A coverage figure on its own invites the wrong game — raise it by testing what
is easy. The pattern this follows (elasticgraph's) makes the number what is
left after decisions: every module below the threshold is listed in
`[tool.coverage.register]` in `pyproject.toml` with a kind and a reason, and
the list is checked both ways.

Fails when:

* the total is below `fail_under` in `pyproject.toml` — the floor only rises;
* a module below the threshold is not in the register — a gap with no reason;
* a registered module has reached the threshold — the entry is stale, and a
  register that only grows stops being read;
* a registered module no longer exists.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THRESHOLD = 60.0
KINDS = {"external", "optional-dependency", "legacy", "debt"}


def fail_under() -> float:
    text = (ROOT / "pyproject.toml").read_text()
    m = re.search(r"^\[tool\.coverage\.report\][^\[]*?^fail_under\s*=\s*([\d.]+)", text, re.M | re.S)
    if not m:
        raise SystemExit("pyproject.toml has no [tool.coverage.report] fail_under")
    return float(m.group(1))


def register() -> dict:
    """{module: (kind, reason)} from `[tool.coverage.register]` in pyproject.toml."""
    if sys.version_info >= (3, 11):
        import tomllib
    else:                                            # pragma: no cover
        import tomli as tomllib
    with open(ROOT / "pyproject.toml", "rb") as fh:
        table = tomllib.load(fh).get("tool", {}).get("coverage", {}).get("register", {})
    return {module: (row.get("kind", ""), row.get("reason", "").strip())
            for module, row in table.items()}


def problems(report: dict, rows: dict, floor: float) -> list:
    out = []
    total = report["totals"]["percent_covered"]
    if total < floor:
        out.append(f"total coverage {total:.2f}% is below the floor {floor}% in pyproject.toml")
    files = {f.replace("\\", "/"): v["summary"]["percent_covered"] for f, v in report["files"].items()}
    for module, pct in sorted(files.items()):
        if pct < THRESHOLD and module not in rows:
            out.append(f"{module} is at {pct:.1f}% with no entry in [tool.coverage.register]")
    for module, (kind, reason) in sorted(rows.items()):
        if kind not in KINDS:
            out.append(f"{module}: kind {kind!r} is not one of {sorted(KINDS)}")
        if not reason:
            out.append(f"{module}: no reason given")
        if module not in files:
            out.append(f"{module} is registered but not measured — removed or renamed?")
        elif files[module] >= THRESHOLD:
            out.append(f"{module} reached {files[module]:.1f}% — remove it from [tool.coverage.register]")
    return out


def main(argv) -> int:
    if not argv:
        print(__doc__); return 2
    report = json.loads(Path(argv[0]).read_text())
    found = problems(report, register(), fail_under())
    for p in found:
        print(f"coverage register: {p}")
    if not found:
        print(f"coverage {report['totals']['percent_covered']:.2f}% (floor {fail_under()}%); "
              f"register holds every module under {THRESHOLD:g}%")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
