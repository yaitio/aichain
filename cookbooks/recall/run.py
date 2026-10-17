"""recall — an assistant that remembers a long conversation, without a vector database.

    python run.py                      replay conversation 0, then answer its questions
    python run.py --check              the same, scored against LoCoMo's own answers
    python run.py --conversation N     another of the ten
    python run.py --model M            another model for the answers
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import ingest
from prereq import require
from receipt import Receipt
from sources import fetch_sources

HERE = Path(__file__).parent


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_argument("--check", action="store_true")
    cli.add_argument("--conversation", type=int, default=0)
    cli.add_argument("--model", default="claude-haiku-5-5")
    args = cli.parse_args()
    require(keys=["ANTHROPIC_API_KEY"],
            packages=[("yait_byheart", "pip install yait-byheart"), ("nltk", "pip install nltk")])

    path = fetch_sources(HERE / "sources.toml", ingest.DATA)["locomo"]
    sample = json.loads(path.read_text())[args.conversation]
    receipt = Receipt()
    with receipt.stage("ingest"):            # nothing is called here: the row stays at zero
        memory, turns = ingest.ingest(sample)
    print(f"{sample['sample_id']}: {turns} turns in {memory.stats.get('turns', turns)} kept, "
          f"saved to data/memory.json")
    receipt.print()
    sys.stdout.flush()                   # before the answering process writes to the same terminal

    one = ingest.DATA / f"sample-{args.conversation}.json"
    one.write_text(json.dumps(sample))
    done = subprocess.run([sys.executable, "answer.py", str(one), "--model", args.model,
                           *(["--check"] if args.check else [])], cwd=HERE)
    sys.exit(done.returncode)


if __name__ == "__main__":
    main()
