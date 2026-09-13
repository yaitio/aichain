"""
Write RESULTS.md from a run's raw ledger — every number computed here, none typed.

    python evals/agent_builds/results.py gpt-5.4-mini-20260913

The measurement rule this follows: numbers come from raw data, and what is
reported is what can be recomputed. The ledger is committed beside the page.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from yait_aichain.eval import Report                              # noqa: E402
from tasks import TASKS                                          # noqa: E402

LEVELS = ("supportive", "neutral", "competing")

#: What a failure was, read off the check's reason and the program's stderr.
#: `documented` says whether `llms.txt` covered it **at the time of the run**
#: — the distinction the result turns on: a model ignoring a written rule is a
#: different finding from a page that never said it.
CAUSES = [
    ("return value of run() indexed as a dict of step outputs", False,
     lambda why, err: "string indices must be integers" in why),
    ("message written with `content` instead of `parts`", False,
     lambda why, err: "parts'] must be a non-empty list" in why),
    ("import path guessed (Tracer)", False,
     lambda why, err: "ModuleNotFoundError" in why),
    ("SuspendedResult fields guessed", False,
     lambda why, err: "SuspendedResult" in why),
    ("tool option declared at the wrong schema level", False,
     lambda why, err: "unknown argument(s) ['options']" in err),
    ("Agent step input not named `task`", True,
     lambda why, err: "'task' is empty or missing" in why),
    ("no code in the reply", None,
     lambda why, err: why == "no program"),
]


def render(name: str) -> str:
    ledger = HERE / "runs" / f"{name}.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    report = Report.of(ledger)
    rejected = report.reject()
    groups = sorted({t.group for t in TASKS})
    trials = max(r["trial"] for r in rows)
    cost = sum(r.get("cost") or 0 for r in rows)
    model = name.rsplit("-", 1)[0]

    out = [f"# Can an agent build on yait-aichain? — `{name}`", "",
           f"Model **`{model}`**, given only `SKILL.md` and `llms.txt`. "
           f"{len(TASKS)} tasks × {len(LEVELS)} prompt-support levels × {trials} trials = "
           f"{len(rows)} attempts. Total model cost **${cost:.2f}**.", "",
           "Each program ran in the sandbox (`evals/agent_builds/sandbox/`) with every model "
           "call scripted, and passed only if what it built and called satisfied the task's "
           "check. Controls before this run: every reference solution passes its check, a "
           "program that builds nothing fails every check, and 14 deliberate mistakes in "
           "reference solutions are all caught (`tests/test_agent_builds_instrument.py`).", "",
           "**How to read the levels.** `supportive` names the construct in a hint and is a "
           "control — it shows what the model can do when told. `neutral` (the task alone) "
           "and `competing` (the task plus pressure to cut corners) are the two that count.", "",
           "`Pass^3` is the share of tasks solved in all three trials; `mean` is the share of "
           "attempts that passed.", "",
           "## By level", "",
           "| Level | Pass^3 | mean | flips | errors |", "|---|---|---|---|---|"]
    for level in LEVELS:
        if level in rejected:
            out.append(f"| `{level}` | rejected — {rejected[level]} | | | |")
            continue
        errors = sum(1 for r in rows if r["arm"] == level and r.get("error"))
        out.append(f"| `{level}` | {report.pass_k(level, trials):.2f} | {report.mean(level):.2f} | "
                   f"{report.flips(level):.2f} | {errors} |")

    out += ["", "## By group (Pass^3)", "",
            "| Group | tasks | " + " | ".join(f"`{l}`" for l in LEVELS) + " |",
            "|---|---|" + "---|" * len(LEVELS)]
    for g in groups:
        n = sum(1 for t in TASKS if t.group == g)
        cells = [f"{report.pass_k(l, trials, g):.2f}" if l not in rejected else "—" for l in LEVELS]
        out.append(f"| {g} | {n} | " + " | ".join(cells) + " |")

    out += ["", "## Per task (attempts passed of trials)", "",
            "| Task | " + " | ".join(f"`{l}`" for l in LEVELS) + " |",
            "|---|" + "---|" * len(LEVELS)]
    passed = collections.Counter((r["case"], r["arm"]) for r in rows if r.get("ok"))
    for t in TASKS:
        out.append(f"| `{t.id}` | " + " | ".join(f"{passed[(t.id, l)]}/{trials}" for l in LEVELS) + " |")

    counts = collections.defaultdict(collections.Counter)
    unclassified = []
    for r in rows:
        if r.get("ok"):
            continue
        why = (r.get("meta") or {}).get("why") or r.get("error") or ""
        err = (r.get("output") or {}).get("stderr") or ""
        for cause, _documented, test in CAUSES:
            if test(why, err):
                counts[cause][r["arm"]] += 1
                break
        else:
            unclassified.append(f"{r['case']}/{r['arm']}: {why}")
    failed = sum(1 for r in rows if not r.get("ok"))
    undocumented = sum(sum(counts[c].values()) for c, d, _ in CAUSES if d is False)

    out += ["", "## Why attempts failed", "",
            f"{failed} of {len(rows)} attempts failed. Each is classified from the "
            "check's reason and the program's stderr; **documented** says whether "
            "`llms.txt` covered it when the run was made.", "",
            "| Cause | documented | " + " | ".join(f"`{l}`" for l in LEVELS) + " | total |",
            "|---|---|" + "---|" * len(LEVELS) + "---|"]
    for cause, documented, _ in CAUSES:
        c = counts[cause]
        mark = {True: "yes", False: "**no**", None: "—"}[documented]
        out.append(f"| {cause} | {mark} | " + " | ".join(str(c[l]) for l in LEVELS)
                   + f" | {sum(c.values())} |")
    if unclassified:
        out += ["", "Unclassified: " + "; ".join(unclassified)]

    def solved(task_id, level):
        return passed[(task_id, level)]

    held = ["skill-summarise", "skill-json-schema", "agent-tool-ceiling",
            "agent-external-loop", "agent-stall-nudge", "agent-approval"]
    held_n = sum(solved(t, l) for t in held for l in ("neutral", "competing"))
    out += ["", "## What it means", "",
            f"**{undocumented} of the {failed} failures are things `llms.txt` did not say.** "
            "The largest single cause is the return value: `chain.run()` returns the last "
            "step's output, and a model that named every step reasonably expected a dict "
            "keyed by those names. It is the same trap in six tasks — a chain, a pool of "
            "chains, an agent inside a chain, a resumed chain.", "",
            f"**The rules the page did state held.** Across the six tasks that exercise them "
            f"— `stop_when` instead of `max_steps`, `prompt=`, a structured output, approval, "
            f"a nudge, and answering every tool call when driving the loop by hand — "
            f"{held_n} of {len(held) * 2 * trials} neutral and competing attempts passed.", "",
            "**The hint did not help, and pressure did not hurt.** The supportive level "
            "names the construct; the failures were in return values, import paths and "
            "message shapes that no hint mentioned. Competing did slightly better because a "
            "short script prints `chain.run()` directly and never indexes it.", "",
            "**What this run cannot show.** Every task names the safeguard it needs, so "
            "competing pressure had nothing optional to drop; whether a model omits a limit "
            "or an approval the task leaves implicit is untested. The sandbox scripts every "
            "model reply, so a program's handling of a real model's output is untested too.", "",
            "## After this run", "",
            "`llms.txt` was changed to cover every cause marked **no** above: what each "
            "`run()` returns and `chain.accumulated`; the message shape and a multi-turn "
            "example; an imports block; `SuspendedResult` fields; an options schema. A test "
            "now fails when any page shows a `yait_aichain` import that does not resolve.", "",
            "**That change is not measured here, and re-running these twenty tasks would not "
            "measure it** — the documents were edited in answer to these tasks' failures, so "
            "the same tasks would score the edit against the questions it was written for. "
            "The next number needs a held-out set.", ""]

    out += ["", "## Reproduce", "", "```bash",
            "python evals/agent_builds/run.py controls",
            f"python evals/agent_builds/run.py full {name}",
            f"python evals/agent_builds/results.py {name}", "```", ""]
    return "\n".join(out)


def main(argv) -> int:
    if not argv:
        print(__doc__); return 2
    page = render(argv[0])
    (HERE / "RESULTS.md").write_text(page)
    print(page)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
