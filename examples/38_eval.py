"""
38_eval.py — Two prompts compared on repeated trials, not on one lucky run.

Keys: none. Runs offline: scripted models play two prompts, and one of them
slips on one case in one trial out of three.
"""

import os
import tempfile
from pathlib import Path

from yait_aichain import Skill
from yait_aichain.eval import Case, Eval, exact
from yait_aichain.testing import ScriptedModel

OUT = Path(os.getenv("AICHAIN_EXAMPLES_DIR", tempfile.gettempdir())) / "aichain-examples"
SUMS = {"2+2": "4", "3*7": "21", "10-4": "6", "9/3": "3", "5+8": "13"}
cases = [Case(id=q, input=q, expect=a) for q, a in SUMS.items()]

# A real arm is Skill(Model("..."), prompt=...). Here each case gets a scripted
# model whose three replies are its three trials; "verbose" is wrong once.
replies = {"terse": {q: [a] * 3 for q, a in SUMS.items()},
           "verbose": {q: [a] * 3 for q, a in SUMS.items()} | {"5+8": ["13", "12", "13"]}}
models = {arm: {q: ScriptedModel(r) for q, r in by_case.items()} for arm, by_case in replies.items()}
prompts = {"terse": "Answer with the number only: {q}",
           "verbose": "Think about it, then answer with the number only: {q}"}


def arm(name):
    return lambda case: Skill(models[name][case.id], prompt=prompts[name]).run(
        variables={"q": case.input})


ledger = OUT / "eval" / "arithmetic.jsonl"
ledger.unlink(missing_ok=True)                       # a fresh run each time
report = Eval(cases, {name: arm(name) for name in prompts}, score=exact(),
              trials=3, out=ledger, verbose=False).run()
print(report.table())
print(f"pass^3: terse {report.pass_k('terse'):.2f}, verbose {report.pass_k('verbose'):.2f}")
print(f"ledger: {ledger}")
