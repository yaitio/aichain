"""survey — a one-page brief on a company from its annual reports and the web,
every claim cited and every disagreement between them listed.

    python run.py              index the filings once, research, write data/report.md
    python run.py --check      the same, then score the figures against the XBRL facts
    python run.py --refresh    download, convert and index again
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import index
import team
from prereq import require
from receipt import Receipt
from sources import fetch_sources

try:
    import tomllib
except ImportError:                       # Python 3.10: aichain installs tomli
    import tomli as tomllib

HERE = Path(__file__).parent
DATA = HERE / "data"
KEY = tomllib.loads((HERE / "answers.toml").read_text())
CHROMA = os.getenv("CHROMA_URL", "http://localhost:8000")


def expected(sources: dict) -> dict:
    """The answer key, read now from the pinned sources: {check id: value}."""
    facts = json.loads(sources["facts"].read_text())["facts"]
    out = {}
    for c in KEY["check"]:
        if "pattern" in c:
            text = json.loads((DATA / "converted" / f"{c['filing']}.json").read_text())["text"]
            out[c["id"]] = float(re.search(c["pattern"], text).group(1).replace(",", ""))
            continue
        rows = [v for tag in c["tags"] for unit in facts["us-gaap"].get(tag, {}).get("units", {}).values()
                for v in unit if v.get("end") == KEY["period"] and v.get("form") == "10-K"
                and v.get("start", "")[:4] < KEY["period"][:4]]
        out[c["id"]] = float(rows[0]["val"])
    return out


def score(sources: dict) -> int:
    report = json.loads((DATA / "report.json").read_text())["figures"]
    passed = 0
    for check, want in expected(sources).items():
        got = report.get(check, {})
        value, source = float(got.get("value") or 0), str(got.get("source", ""))
        close = want and abs(value - want) / want <= KEY["tolerance"]
        ok = bool(close) and "10-K" in source
        passed += ok
        print(f"  {'PASS' if ok else 'FAIL'} {check}: {value:,.0f} from {source!r} (filing: {want:,.0f})")
    results = re.search(r"## Results\s*(.*?)(?=\n## |\Z)", (DATA / "report.md").read_text(), re.S)
    # A claim is a bullet, whose citation ends it, or a sentence of plain prose.
    body = results.group(1).strip() if results else ""
    sentences = [line for line in body.splitlines() if line.lstrip().startswith(("-", "*"))] or \
        [s for s in re.split(r"(?<=[.!?])\s+", body) if s]
    uncited = [s for s in sentences if not re.search(r"Item \d|https?://", s)]
    cited = bool(sentences) and not uncited
    print(f"  {'PASS' if cited else 'FAIL'} results-cited: "
          f"{len(sentences) - len(uncited)}/{len(sentences)} claims cite an Item or a URL")
    return passed + cited


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_argument("--check", action="store_true")
    cli.add_argument("--refresh", action="store_true")
    args = cli.parse_args()
    require(keys=["ANTHROPIC_API_KEY", "GOOGLE_AI_API_KEY", "SEC_USER_AGENT", tuple(team.SEARCH.values())],
            hints={"SEC_USER_AGENT": 'SEC asks for a name and an email: export SEC_USER_AGENT="Jane Doe jane@example.com"'},
            packages=[("markitdown", "pip install markitdown")],
            services=[("Chroma", f"{CHROMA}/api/v2/heartbeat",
                       "pip install chromadb && chroma run --path data/chroma")])
    if args.refresh:
        for item in DATA.glob("*"):
            if item.name != "chroma":
                shutil.rmtree(item) if item.is_dir() else item.unlink()
    for stale in ("report.md", "report.json"):
        (DATA / stale).unlink(missing_ok=True)

    receipt = Receipt()
    sources = fetch_sources(HERE / "sources.toml", DATA, user_agent=os.environ["SEC_USER_AGENT"],
                            min_interval=0.2)
    kb, _ = index.build(sources, receipt, refresh=args.refresh)
    hook = receipt.hook()
    with receipt.stage("research"):
        result = team.build_lead(kb, hooks=[hook]).run(
            team.BRIEF.format(company=KEY["company"], period=KEY["period"]))
    print((DATA / "report.md").read_text() if (DATA / "report.md").exists()
          else f"no report: {result.stopped_by} {result.error or ''}")
    receipt.print()
    if args.check:
        if not (DATA / "report.json").exists():
            sys.exit(1)
        passed = score(sources)
        print(f"\n{passed}/{len(KEY['check']) + 1} checks passed")
        sys.exit(0 if passed == len(KEY["check"]) + 1 else 1)


if __name__ == "__main__":
    main()
