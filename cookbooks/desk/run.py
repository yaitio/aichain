"""desk — a support agent that refunds small, asks a manager for big, in one process.

    python run.py             the four scenarios, the manager answering in code
    python run.py --check     the same, scored against answers.toml
    python run.py --model M   another model for the agent

simulate.py plays the same scenarios as two serverless handlers in two processes.
"""

import argparse
import os
import shutil
import sqlite3
import sys
from pathlib import Path

from yait_aichain.state import RunContext, SuspendedResult, using

import desk
import setup
from prereq import require
from receipt import Receipt

try:
    import tomllib
except ImportError:                       # Python 3.10: aichain installs tomli
    import tomli as tomllib

HERE = Path(__file__).parent
SCENARIOS = tomllib.loads((HERE / "answers.toml").read_text())["scenario"]


def fresh_shop() -> None:
    """Every run starts from the same shop: fifty orders, nothing refunded, nothing parked."""
    setup.build()
    shutil.rmtree(setup.DATA / "runs", ignore_errors=True)


def refunded(order: str) -> bool:
    with sqlite3.connect(setup.DATA / "shop.db") as db:
        return bool(db.execute("SELECT refunded FROM orders WHERE id = ?", (order,)).fetchone()[0])


def score(outcomes: dict) -> int:
    """One line per check, PASS or FAIL and why; the number that passed.
    *outcomes*: scenario id → {"parked", "stopped_by", "reply"}."""
    passed = 0
    for s in SCENARIOS:
        got = outcomes[s["id"]]
        wrong = [what for what, bad in (
            (f"refunded is {refunded(s['order'])}", refunded(s["order"]) != s["refunded"]),
            (f"parked is {got['parked']}", got["parked"] != s["parked"]),
            (f"stopped by {got['stopped_by']}", got["stopped_by"] != "answered"),
            (f"the reply does not say {s.get('mentions')!r}",
             bool(s.get("mentions")) and s.get("mentions", "") not in str(got["reply"]).lower()),
        ) if bad]
        passed += not wrong
        print(f"  {'FAIL' if wrong else 'PASS'} {s['id']}" + (f": {'; '.join(wrong)}" if wrong else ""))
    stored = "".join(p.read_text() for p in (setup.DATA / "runs").rglob("*") if p.is_file())
    leaked = os.environ["ANTHROPIC_API_KEY"] in stored or "sk-" in stored
    passed += not leaked
    print(f"  {'FAIL' if leaked else 'PASS'} no-key-in-store" + (": a key is in data/runs" if leaked else ""))
    return passed


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_argument("--check", action="store_true")
    cli.add_argument("--model", default=desk.MODEL)
    args = cli.parse_args()
    require(keys=["ANTHROPIC_API_KEY"])
    desk.MODEL = args.model
    fresh_shop()

    receipt, outcomes = Receipt(), {}
    with using(RunContext(tenant="acme")):
        agent = desk.build_agent(hooks=[receipt.hook()])
        for s in SCENARIOS:
            print(f"\ncustomer: {s['message']}")
            with receipt.stage(s["id"]):
                result = agent.run(s["message"])
                parked = isinstance(result, SuspendedResult)
                if parked:
                    print(f"  parked, run {result.run_id}: {result.awaiting.get('reason')}")
                    print(f"  manager: {s['approval']}")
                    result = agent.resume(result.run_id, signal=s["approval"])
            print(f"desk: {result.output}")
            outcomes[s["id"]] = {"parked": parked, "stopped_by": result.stopped_by, "reply": result.output}
    receipt.print()
    if args.check:
        passed = score(outcomes)
        print(f"\n{passed}/{len(SCENARIOS) + 1} checks passed")
        sys.exit(0 if passed == len(SCENARIOS) + 1 else 1)


if __name__ == "__main__":
    main()
