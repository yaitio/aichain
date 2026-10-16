"""
Reconcile — one task, four agent strategies, one answer key.

The job is the one every company already has: check a supplier's invoices
against the contract and the delivery notes, and report what is wrong. The
faults are planted by ``generate.py``, so scoring is exact and free — no judge
model, no hand labelling, and nothing to argue about afterwards.

Four approaches answer the same question through the same tools:

    agile       Agent(model, tools)                    — the loop, deciding each step
    waterfall   Agent(..., mode="waterfall")           — a plan written at step 0, held
    team        Agent(..., team={"extend": True})      — hires workers as it needs them
    scaffold    a coordinator model composes a pipeline — the hand-built control

Every finding arrives through ``report_finding``, so the *actions* are the
answer and no output format has to be parsed. Every model call is on the
receipt, so the comparison is in money, not vibes.

    export ANTHROPIC_API_KEY=...
    python reconcile.py --cases 3
    python reconcile.py --cases 3 --modes team,scaffold --model claude-haiku-5-5
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time

from yait_aichain import Agent, Budget, Model, Pool, Skill
from yait_aichain.agent import step_count, token_budget
from yait_aichain.tools import Tool

from receipt import Receipt

from generate import KINDS, make_case, score

MODEL = "claude-sonnet-5-5"

# The roster the scaffold coordinator picks from. Prices per 1M tokens, from
# the aichain registry, 2026-10-17.
ROSTER = {
    "claude-opus-5-5":   {"tier": "frontier", "in": 4.0, "out": 20.0, "note": "deepest reasoning"},
    "claude-sonnet-5-5": {"tier": "mid",      "in": 2.0, "out": 10.0, "note": "strong all-rounder"},
    "claude-haiku-5-5":  {"tier": "cheap",    "in": 0.1, "out": 0.5,  "note": "fast and very cheap"},
}
COORDINATOR = "claude-sonnet-5-5"

# The same leash for everyone, so nobody is compared at a different ceiling.
# All three are backstops: the ordinary exit is the model answering.
MAX_TOKENS = 150_000
MAX_STEPS  = 30
MAX_COST   = 0.60

TASK = f"""\
You are reconciling a supplier's billing for one client.

Read every document available to you and find EVERY discrepancy. Report each
one by calling report_finding — a discrepancy you describe in prose but never
report does not count.

Fault kinds, use exactly these labels:
""" + "\n".join(f"  {k:<18} {d}" for k, (_, d) in KINDS.items()) + """

Rules:
  - `document` is the invoice id the fault is in, e.g. INV-2026-1042.
  - `line` is the line number shown in the invoice's first column. Omit it for
    a fault in the header (currency, service period, the stated total).
  - Report each distinct fault once. Wrong reports count against you, so do not
    report a line unless you can state the two numbers that disagree.
"""


# ── Tools — the same three for every strategy ────────────────────────────────

class ListDocuments(Tool):
    name        = "list_documents"
    description = "List the names of every document available for this reconciliation."
    parameters  = {"type": "object", "properties": {}}
    risk        = "read"

    def __init__(self, case): self._case = case

    def run(self, options=None) -> list:
        return sorted(self._case.documents)


class ReadDocument(Tool):
    name        = "read_document"
    description = "Read one document in full, by its name from list_documents."
    parameters  = {"type": "object", "properties": {
        "name": {"type": "string", "description": "Document name from list_documents."},
    }, "required": ["name"]}
    risk        = "read"

    def __init__(self, case): self._case = case

    def run(self, name: str, options=None) -> str:
        doc = self._case.documents.get(str(name).strip())
        return doc if doc else f"No document named {name!r}. Call list_documents."


class ReportFinding(Tool):
    name        = "report_finding"
    description = "Report one discrepancy. Call once per distinct fault."
    parameters  = {"type": "object", "properties": {
        "kind":     {"type": "string", "enum": list(KINDS)},
        "document": {"type": "string", "description": "Invoice id the fault is in."},
        "line":     {"type": "integer", "description": "Invoice line number; omit for a header fault."},
        "detail":   {"type": "string", "description": "The two values that disagree."},
    }, "required": ["kind", "document", "detail"]}
    risk        = "read"          # it writes to a notepad, not to the world

    def __init__(self, sink: list): self._sink = sink

    def run(self, kind: str, document: str, detail: str = "",
            line=None, options=None) -> str:
        self._sink.append({"kind": kind, "document": document,
                           "line": line, "detail": detail})
        return f"recorded ({len(self._sink)} so far)"


# ── The Agent, in three configurations ───────────────────────────────────────

#: The concrete ask. The standing brief goes into ``instructions`` instead,
#: where it becomes part of the stable prefix and is cached from the second
#: call onward — a split the previous agent could not make, because it rebuilt
#: its whole prompt every step.
ASK = "Reconcile this supplier's billing and report every discrepancy you find."


def run_agent(case, variant: str, model: str) -> dict:
    found: list = []
    receipt = Receipt()
    agent = Agent(
        # 8000, matching what the scaffold's roles get. At the provider default
        # a long structured reply hits the cap mid-string, and a truncated
        # answer is not a short answer — it is an unparseable one.
        Model(model, options={"max_tokens": 8000}),
        tools        = [ListDocuments(case), ReadDocument(case), ReportFinding(found)],
        instructions = TASK,
        mode         = "waterfall" if variant == "waterfall" else "agile",
        # It may hire workers of its own as it goes.
        team         = {"extend": True} if variant == "team" else None,
        stop_when    = [step_count(MAX_STEPS), token_budget(MAX_TOKENS)],
        max_cost     = MAX_COST,
        hooks        = [receipt.hook()],
    )
    t0     = time.monotonic()
    result = agent.run(ASK)

    # How the run ended is now a first-class answer, not something to infer.
    # "answered" and "hit a ceiling" used to look identical from the outside,
    # which is exactly how four zero-scoring traces became indistinguishable.
    note = "" if result.stopped_by == "answered" else f"stopped by {result.stopped_by}"
    if unfinished := [e for e in result.journal if e.get("outcome") == "skipped"]:
        note += f", {len(unfinished)} plan item(s) outstanding"

    # The shape it built, in the primitives it built it from — the thing worth
    # watching now that the loop can fan out. No score shows whether `pool` is
    # actually being reached for.
    seen: list = []
    for e in result.journal:
        # The tool is in `action`, which the journal keeps as the call's JSON
        # with a fingerprint after it; `intent` is set only for a plan goal.
        a = e.get("action")
        named = (a.get("name") if isinstance(a, dict)
                 else (re.search(r'"name":\s*"([^"]+)"', a or "") or [None, None])[1])
        t = e.get("intent") or named or "?"
        if not seen or seen[-1] != t:
            seen.append(t)
    shape = "→".join(seen) + f"  {result.steps_taken} actions"

    return {"found": found, "receipt": receipt, "seconds": time.monotonic() - t0,
            "steps": result.steps_taken, "ok": bool(result),
            "tokens": result.tokens_used, "shape": shape, "note": note}


# ── The scaffold ─────────────────────────────────────────────────────────────
#
# A compact cousin of cookbook/scaffold — a coordinator reads the task *and the
# price list* and composes the team, then the team runs. It does not use Agent
# at all: it is Model + Skill + Pool, which is why it belongs in this comparison
# as a fourth strategy rather than a fourth mode.

PLAN_SCHEMA = {
    "type": "json_schema", "name": "scaffold", "strict": True,
    "schema": {"type": "object", "properties": {
        "reasoning": {"type": "string", "description": "why this shape fits — two sentences"},
        "reviewers": {"type": "array", "description": "1-3 reviewers, each with a distinct remit",
                      "items": {"type": "object", "properties": {
                          "model": {"type": "string"}, "remit": {"type": "string"}},
                          "required": ["model", "remit"], "additionalProperties": False}},
        "consolidator": {"type": "object", "description": "merges the reviews into the final list",
                         "properties": {"model": {"type": "string"}},
                         "required": ["model"], "additionalProperties": False},
    }, "required": ["reasoning", "reviewers", "consolidator"], "additionalProperties": False},
}

FINDINGS_SCHEMA = {
    "type": "json_schema", "name": "findings", "strict": True,
    "schema": {"type": "object", "properties": {
        "findings": {"type": "array", "items": {"type": "object", "properties": {
            "kind":     {"type": "string", "enum": list(KINDS)},
            "document": {"type": "string"},
            "line":     {"type": ["integer", "null"]},
            "detail":   {"type": "string"},
        }, "required": ["kind", "document", "line", "detail"], "additionalProperties": False}},
    }, "required": ["findings"], "additionalProperties": False},
}


class Ask(Tool):
    """Pool runner, so the reviewers work at the same time rather than in turn."""
    name       = "ask"
    parameters = {"type": "object", "properties": {
        "model": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["model", "prompt"]}

    def __init__(self, receipt, budget): self._receipt, self._budget = receipt, budget

    def run(self, model: str, prompt: str, options=None) -> dict:
        skill = Skill(
            model  = Model(model, options={"max_tokens": 8000}),
            input  = {"messages": [{"role": "user", "parts": [prompt]}]},
            output = {"format": FINDINGS_SCHEMA},
            hooks  = [self._receipt.hook()],
            max_cost = self._budget,
        )
        return skill.run()


def run_scaffold(case, model: str) -> dict:
    receipt = Receipt()
    budget  = Budget(MAX_COST)            # one ceiling across every call of the run
    t0      = time.monotonic()
    bundle = "\n\n".join(f"=== {n} ===\n{t}" for n, t in sorted(case.documents.items()))

    # ── 0. Design the team for this case ──────────────────────────────────
    menu = "\n".join(f"  {m}: {v['tier']}, ${v['in']}/${v['out']} per 1M — {v['note']}"
                     for m, v in ROSTER.items())
    plan = Skill(
        model  = Model(COORDINATOR, options={"max_tokens": 4000}),
        input  = {"messages": [{"role": "user", "parts": [
            f"Design a review team for this job.\n\n{TASK}\n\n"
            f"The job has {len(case.documents)} documents.\n\n"
            f"MODELS AVAILABLE:\n{menu}\n\n"
            "Split the work by remit — for example one reviewer on arithmetic "
            "inside each document, one on cross-checks against the contract and "
            "the delivery notes. Use the cheapest model that can do each remit."]}]},
        output = {"format": PLAN_SCHEMA},
        hooks  = [receipt.hook()], max_cost = budget,
    ).run()

    # ── 1. Reviewers, in parallel, each on its own remit ──────────────────
    # The schema asks for 1-3; a coordinator that returns none would otherwise
    # reach Pool with max_flows=0 and fail on an empty fan-out rather than here.
    if not plan["reviewers"]:
        raise ValueError("coordinator designed a team with no reviewers")
    reviews = Pool(
        Ask(receipt, budget),
        items = [{"model": r["model"],
                  "prompt": f"{TASK}\n\nYOUR REMIT — look only for this: {r['remit']}\n\n"
                            f"Instead of calling report_finding, return the findings as JSON.\n\n{bundle}"}
                 for r in plan["reviewers"]],
        max_flows = len(plan["reviewers"]),
    ).run()

    # ── 2. Consolidate — de-duplicate and drop what cannot be substantiated ─
    pooled = [f for r in reviews if r for f in r.get("findings", [])]
    merged = Skill(
        model  = Model(plan["consolidator"]["model"], options={"max_tokens": 8000}),
        input  = {"messages": [{"role": "user", "parts": [
            f"{TASK}\n\nYour reviewers reported the findings below. Merge them: drop "
            "duplicates, and drop any you cannot confirm against the documents. "
            "Return the surviving list.\n\n"
            f"REPORTED:\n{json.dumps(pooled, indent=1)}\n\n{bundle}"]}]},
        output = {"format": FINDINGS_SCHEMA},
        hooks  = [receipt.hook()], max_cost = budget,
    ).run()

    return {"found": merged.get("findings", []), "receipt": receipt,
            "seconds": time.monotonic() - t0,
            "steps": len(plan["reviewers"]) + 2, "ok": True,
            "tokens": sum(r["tokens"] for r in receipt.rows.values()), "note": "",
            "shape": f"{len(plan['reviewers'])} reviewer(s)→consolidator  "
                     f"{len(plan['reviewers']) + 2} calls"}


# ── Harness ──────────────────────────────────────────────────────────────────

RUNNERS = {
    "agile":     lambda case, m: run_agent(case, "agile",     m),
    "waterfall": lambda case, m: run_agent(case, "waterfall", m),
    "team":      lambda case, m: run_agent(case, "team",      m),
    "scaffold":  run_scaffold,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=3)
    ap.add_argument("--seed",  type=int, default=0)
    ap.add_argument("--invoices", type=int, default=3)
    ap.add_argument("--defects",  type=int, default=4)
    ap.add_argument("--modes", default=",".join(RUNNERS))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--detail", action="store_true",
                    help="add the diagnostic table: recall, precision, label "
                         "agreement, cost per fault, and recall by fault depth")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    cases = [make_case(args.seed + i, args.invoices, args.defects)
             for i in range(args.cases)]
    print(f"{len(cases)} case(s), {args.defects} planted faults each, "
          f"model {args.model}\n")

    results: dict = {m: [] for m in modes}
    for case in cases:
        print(f"case seed={case.seed}  depths={case.depth_counts()}")
        for mode in modes:
            try:
                run = RUNNERS[mode](case, args.model)
            except Exception as exc:
                print(f"  {mode:<10} ERROR {type(exc).__name__}: {exc}")
                continue
            s = score(case, run["found"])
            rows = run["receipt"].rows.values()
            s.update(cost=run["receipt"].total(), tokens=sum(r["tokens"] for r in rows),
                     calls=sum(r["calls"] for r in rows), seconds=run["seconds"],
                     steps=run["steps"])
            results[mode].append(s)
            print(f"  {mode:<10} {'solved' if s['solved'] else '  --  '}  "
                  f"{s['found']}/{s['planted']} found, "
                  f"{s['false_positives']} false  "
                  f"${s['cost']:.4f}  {run.get('tokens', 0):,} tok  "
                  f"{s['seconds']:.0f}s"
                  + (f"  [{run['shape']}]" if run.get("shape") else "")
                  + (f"  ⚠ {run['note']}" if run.get("note") else ""))
        print()

    # `solved` answers "can I act on this without re-checking it" — all planted
    # faults found and nothing invented. It is deliberately all-or-nothing, but
    # all-or-nothing throws away the difference between three of four and none,
    # so `found` carries the partial credit beside it.
    #
    # `false` stays its own column rather than being folded into a ratio. A
    # strategy that flags every line reaches full `found`; the only thing that
    # exposes it is the count of what it invented, and a reader should not have
    # to infer that from a precision figure.
    print(f"{'mode':<11}{'solved':>9}{'found':>9}{'false':>7}"
          f"{'$/case':>10}{'sec':>7}")
    for mode in modes:
        rs = results[mode]
        if not rs:
            continue
        avg     = lambda k: statistics.mean(r[k] for r in rs)
        solved  = sum(1 for r in rs if r["solved"])
        found   = sum(r["found"]   for r in rs)
        planted = sum(r["planted"] for r in rs)
        false   = sum(r["false_positives"] for r in rs)
        print(f"{mode:<11}{f'{solved}/{len(rs)}':>9}{f'{found}/{planted}':>9}"
              f"{false:>7}{avg('cost'):>10.4f}{avg('seconds'):>7.0f}")

    if not args.detail:
        return

    # Ratios are pooled over all cases, not averaged over per-case ratios.
    # Averaging hid the thing that mattered: a case where a mode reported
    # nothing scored precision 1.0, because an empty report has no wrong
    # answers in it. Waterfall came out at 0.80 on the strength of the four
    # cases where it stayed silent; pooled, it is 0.50 on two reports.
    # The raw counts are printed beside the ratios for the same reason —
    # a ratio computed from two observations should not be able to pass for
    # a result.
    print(f"\n{'mode':<11}{'found':>10}{'reported':>10}{'recall':>8}"
          f"{'precision':>11}{'kind':>7}{'$/fault':>10}{'calls':>7}")
    for mode in modes:
        rs = results[mode]
        if not rs:
            continue
        found    = sum(r["found"]    for r in rs)
        planted  = sum(r["planted"]  for r in rs)
        reported = sum(r["reported"] for r in rs)
        agreed   = sum(r["kind_agreement"] * r["found"] for r in rs)
        cost     = sum(r["cost"] for r in rs)
        ratio    = lambda n, d: f"{n / d:>7.2f}" if d else f"{'n/a':>7}"
        print(f"{mode:<11}{f'{found}/{planted}':>10}{reported:>10}"
              f"{ratio(found, planted):>8}{ratio(found, reported):>11}"
              f"{ratio(agreed, found):>7}"
              f"{(cost / found if found else float('nan')):>10.4f}"
              f"{statistics.mean(r['calls'] for r in rs):>7.1f}")

    depths = sorted({d for rs in results.values() for r in rs for d in r["by_depth"]})
    if depths:
        print(f"\nrecall by depth (found/planted)\n{'mode':<11}"
              + "".join(f"{d:>12}" for d in depths))
        for mode in modes:
            rs = results[mode]
            if not rs:
                continue
            row = ""
            for d in depths:
                hit = sum(r["by_depth"].get(d, (0, 0))[0] for r in rs)
                tot = sum(r["by_depth"].get(d, (0, 0))[1] for r in rs)
                row += f"{f'{hit}/{tot}':>12}" if tot else f"{'-':>12}"
            print(f"{mode:<11}{row}")


if __name__ == "__main__":
    main()
