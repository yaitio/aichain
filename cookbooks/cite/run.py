"""cite — answers from three EU regulations and a scanned page, with the article that says so.

    python run.py                    the questions of answers.toml
    python run.py "your question"    one question of your own
    python run.py --check            the same, scored against answers.toml
    python run.py --refresh          convert and index again from scratch
    python run.py --model M          answer with another model
    python run.py --no-rerank        no Cohere key: take the store's top six
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from yait_aichain.tools import rerank

import answer
import index
from prereq import require
from receipt import Receipt
from sources import fetch_sources

try:
    import tomllib
except ImportError:                       # Python 3.10: aichain installs tomli
    import tomli as tomllib

HERE = Path(__file__).parent
DATA = HERE / "data"
CHROMA = os.getenv("CHROMA_URL", "http://localhost:8000")


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_argument("question", nargs="?")
    cli.add_argument("--check", action="store_true")
    cli.add_argument("--refresh", action="store_true")
    cli.add_argument("--model", default=answer.ANSWER_MODEL)
    cli.add_argument("--no-rerank", action="store_true")
    args = cli.parse_args()

    require(keys=["ANTHROPIC_API_KEY", "GOOGLE_AI_API_KEY"] + ([] if args.no_rerank else ["COHERE_API_KEY"]),
            hints={"COHERE_API_KEY": "set it, or run with --no-rerank"},
            packages=[("markitdown", "pip install 'markitdown[pdf]'")],
            services=[("Chroma", f"{CHROMA}/api/v2/heartbeat",
                       "pip install chromadb && chroma run --path data/chroma")])
    if args.refresh:                      # everything but the running store's own files
        for item in DATA.glob("*"):
            if item.name != "chroma":
                shutil.rmtree(item) if item.is_dir() else item.unlink()

    receipt = Receipt()
    sources = fetch_sources(HERE / "sources.toml", DATA / "sources")
    kb, parts = index.build(sources, receipt, refresh=args.refresh)
    texts = {(p["regulation"], p["article"]): p["text"] for p in parts}
    reranker = None if args.no_rerank else rerank("cohere", model="rerank-v3.5")

    key = tomllib.loads((HERE / "answers.toml").read_text())
    checks = [{"id": "yours", "question": args.question}] if args.question else key["check"]
    passed = 0
    with receipt.stage("answer"):
        for check in checks:
            result = answer.answer(kb, check["question"], receipt, args.model, reranker)
            print(f"\n{check['question']}\n  {result.get('answer') or '(refused)'}")
            for c in result.get("citations") or []:
                print(f"  — {c['regulation']}, {c['article']}: “{c['quote'][:140]}”")
            if args.check and not args.question:
                ok, why = answer.verdict(check, result, texts)
                passed += ok
                print(f"  {'PASS' if ok else 'FAIL'} {check['id']}" + (f": {why}" if why else ""))
    receipt.print()
    if args.check and not args.question:
        score = passed / len(checks)
        print(f"\nscore {passed}/{len(checks)} = {score:.2f} (threshold {key['threshold']})")
        sys.exit(0 if score >= key["threshold"] else 1)


if __name__ == "__main__":
    main()
