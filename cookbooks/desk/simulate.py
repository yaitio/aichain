"""The four scenarios through handlers.py, each handler in a process of its own.

    python simulate.py                       every scenario, start then approve
    python simulate.py --check               the same, scored
    python simulate.py start SCENARIO        only the first handler; prints the run id
    python simulate.py approve SCENARIO RUN  only the second, for a run parked earlier

Only the JSON event crosses into a handler's process, so what the two share is
data/ and nothing else — the way two Lambda invocations would.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import run
from prereq import require
from receipt import Receipt

HERE = Path(__file__).parent
TENANT = "acme"


def invoke(handler: str, event: dict) -> dict:
    code = (f"import json, sys; from handlers import {handler}; "
            f"print(json.dumps({handler}(json.loads(sys.argv[1]))))")
    done = subprocess.run([sys.executable, "-c", code, json.dumps(event)],
                          cwd=HERE, capture_output=True, text=True)
    if done.returncode:
        raise SystemExit(f"{handler} failed:\n{done.stderr}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def scenario(name: str) -> dict:
    return next(s for s in run.SCENARIOS if s["id"] == name)


def approve_event(s: dict, run_id: str) -> dict:
    return {"tenant": TENANT, "run_id": run_id, **s["approval"]}


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_argument("--check", action="store_true")
    cli.add_argument("step", nargs="*", help="start SCENARIO | approve SCENARIO RUN_ID")
    args = cli.parse_args()
    require(keys=["ANTHROPIC_API_KEY"])
    if args.step[:1] == ["start"]:
        run.fresh_shop()
        print(json.dumps(invoke("start", {"tenant": TENANT, "message": scenario(args.step[1])["message"]})))
        return
    if args.step[:1] == ["approve"]:
        print(json.dumps(invoke("approve", approve_event(scenario(args.step[1]), args.step[2]))))
        return

    run.fresh_shop()
    receipt, outcomes = Receipt(), {}
    for s in run.SCENARIOS:
        with receipt.stage(s["id"]):
            first = invoke("start", {"tenant": TENANT, "message": s["message"]})
            parked = first["status"] == "parked"
            print(f"\n{s['id']}: start → {first['status']}" + (f" ({first['run_id']})" if parked else ""))
            last = invoke("approve", approve_event(s, first["run_id"])) if parked else first
            for step in (first, last) if parked else (first,):
                if step.get("cost") is not None or step.get("tokens"):
                    receipt.add(step)
            print(f"  desk: {last['reply']}")
            outcomes[s["id"]] = {"parked": parked, "stopped_by": last["stopped_by"], "reply": last["reply"]}
    receipt.print()
    if args.check:
        passed = run.score(outcomes)
        print(f"\n{passed}/{len(run.SCENARIOS) + 1} checks passed")
        sys.exit(0 if passed == len(run.SCENARIOS) + 1 else 1)


if __name__ == "__main__":
    main()
