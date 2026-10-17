"""brief — a ten-page literature brief on twelve papers, longer than any one answer.

    python run.py              summarise, outline, write data/brief.md
    python run.py --check      the same, then check it against answers.toml
    python run.py --one-call   ask for the whole brief in one call, to see it cut off
"""

import argparse
import json
import sys
from pathlib import Path

from yait_aichain import Model, Pool, Skill

import write
from convert import convert
from prereq import require
from receipt import Receipt
from sources import fetch_sources

try:
    import tomllib
except ImportError:                       # Python 3.10: aichain installs tomli
    import tomli as tomllib

HERE = Path(__file__).parent
DATA = HERE / "data"
SUMMARY_MODEL = "claude-haiku-5-5"
TOPIC = "how to measure whether an LLM agent is reliable"
PAPER_CHARS = 150_000                     # the body of the longest paper; appendices beyond it
SUMMARY = """Summarise this paper in at most 300 words, under exactly these headings:
**Question**, **Method**, **Result**, **Limits**. Numbers as the paper gives them.

{paper}"""
OUTLINE = {"type": "object", "required": ["sections"], "properties": {"sections": {
    "type": "array", "minItems": 6, "maxItems": 10, "items": {
        "type": "object", "required": ["title", "papers"], "properties": {
            "title": {"type": "string"}, "papers": {"type": "array", "items": {"type": "string"}}}}}}}


def summaries(papers: dict, hook) -> dict:
    """{arXiv id: summary}, six at a time, by a cheap model; kept in data/summaries.json."""
    saved = DATA / "summaries.json"
    if saved.exists():
        return json.loads(saved.read_text())
    texts = [convert.run(path=str(path))["text"][:PAPER_CHARS] for path in papers.values()]
    done = Pool(Skill(Model(SUMMARY_MODEL), prompt=SUMMARY, hooks=[hook]),
                items=[{"paper": t} for t in texts], max_flows=6).run()
    out = dict(zip(papers, done.output))
    saved.write_text(json.dumps(out, indent=1))
    return out


def outline(found: dict, hook) -> list[dict]:
    menu = "\n\n".join(f"[{p}]\n{s}" for p, s in found.items())
    skill = Skill(Model(write.WRITER_MODEL), hooks=[hook], output={"format": {"type": "json_schema", "schema": OUTLINE}},
                  prompt="Plan a literature brief on {topic} from these twelve summaries: 6 to 10 "
                         "sections, each with a title and the arXiv ids of the papers it draws on. "
                         "Every paper belongs to at least one section.\n\n{menu}")
    return skill.run(variables={"topic": TOPIC, "menu": menu})["sections"]


def check(brief: str, spent: int, ids: list[str]) -> int:
    body = brief.split("\n## References")[0]
    cap = Model(write.WRITER_MODEL).effective_options["max_tokens"]
    missing = [p for p in ids if f"[{p}]" not in body]
    results = [(f"longer than one answer: {spent:,} output tokens against the writer's cap of {cap:,}",
                spent > cap),
               ("no section was cut at max_tokens", True),     # write() fails the run otherwise
               (f"all {len(ids)} papers cited" + (f"; missing {', '.join(missing)}" if missing else ""),
                not missing)]
    for what, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'} {what}")
    return sum(ok for _, ok in results)


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_argument("--check", action="store_true")
    cli.add_argument("--one-call", action="store_true")
    args = cli.parse_args()
    require(keys=["ANTHROPIC_API_KEY"], packages=[("markitdown", "pip install 'markitdown[pdf]'")])
    listed = tomllib.loads((HERE / "sources.toml").read_text())["source"]
    papers = fetch_sources(HERE / "sources.toml", DATA / "sources", min_interval=3.0)
    receipt = Receipt()
    hook = receipt.hook()

    with receipt.stage("summarise"):
        found = summaries(papers, hook)
    if args.one_call:
        with receipt.stage("one call"):
            skill = Skill(Model(write.WRITER_MODEL), hooks=[hook], prompt=(
                "Write a ten-page literature brief, about 9,000 words, on {topic} from these "
                "summaries, citing each paper as [arXiv id].\n\n{menu}"))
            text = skill.run(variables={"topic": TOPIC, "menu": json.dumps(found, indent=1)})
        print(f"{len(text.split()):,} words; cut at max_tokens: {skill.last_truncated}; "
              f"ends: …{text[-120:]!r}")
        receipt.print()
        return

    with receipt.stage("outline"):
        plan = outline(found, hook)
    with receipt.stage("write"):
        sections, spent = write.write(TOPIC, plan, found, hooks=[hook])
    references = "\n".join(f"- [{s['id']}] {s['title']}. https://arxiv.org/abs/{s['file'][:-4]}"
                           for s in listed)
    brief = f"# {TOPIC.capitalize()}: a brief\n\n" + "\n\n".join(sections) + f"\n\n## References\n\n{references}\n"
    (DATA / "brief.md").write_text(brief)
    print(f"data/brief.md: {len(plan)} sections, {len(brief.split()):,} words, {spent:,} output tokens")
    receipt.print()
    if args.check:
        passed = check(brief, spent, list(papers))
        need = tomllib.loads((HERE / "answers.toml").read_text())["threshold"]
        print(f"\n{passed}/3 checks passed")
        sys.exit(0 if passed >= need else 1)


if __name__ == "__main__":
    main()
