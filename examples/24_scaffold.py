"""
24 · Scaffold — the model designs the team, then runs it
========================================================

The sixth cell of the design grid::

    Agent(model, tools, mode="waterfall", team="auto")

``mode="waterfall"`` makes the agent write a plan at step 0 and hold to it.
``team="auto"`` gives it ``delegate``: workers it describes itself, each
running its own conversation with the parent's tools. Together they are the
scaffold pattern — the coordinator composes the crew for *this* task, the
crew does the work, the coordinator synthesizes. (Sakana's Fugu ships this
shape as a closed endpoint; here every seam of it is visible and priced.)

One more split carries the economics: **think while designing, act without
deliberating.** ``planner_model`` gets reasoning for the one turn where it
pays — the plan — and the execution turns run fast and cheap. Measured on a
dialogue benchmark, this split beat both the always-thinking and the
never-thinking configurations of the same model.

Run it::

    export OPENAI_API_KEY=...
    python examples/24_scaffold.py

What to check in the output
---------------------------
* The plan comes first — the team design, before any work.
* ``delegate`` entries: one worker per report, each described by the model.
* Every delegation result carries ``model_claim`` evidence — a worker's
  report is the worker's word, and the journal names that instead of letting
  it pass as verified fact.
* ``result.cost`` includes the workers' spend: a child's bill rolls up.

Required env vars:
    OPENAI_API_KEY
"""

import textwrap

MODEL = "gpt-5.6-terra"          # any provider you have a key for

from yait_aichain import Agent, Model
from yait_aichain.agent import step_count, cost_budget
from yait_aichain.tools import Tool

# ── Three quarterly reports — the bulk the coordinator should NOT read ────────

REPORTS = {
    "north": """
        Q3 in the north region opened weak: the flagship store lost four
        weekends to flooding, and foot traffic never fully recovered, ending
        11% below Q2. Online partially compensated with a 23% jump after the
        loyalty relaunch, but margin took a hit from aggressive couponing —
        contribution margin fell 2.1pp. Inventory is the quiet worry: winter
        stock arrived six weeks early and is eating warehouse capacity, and
        two suppliers have already signalled January price increases around
        4%. Headcount is stable; one store manager vacancy open since July.
        """,
    "south": """
        The south region delivered the strongest quarter in two years.
        Same-store sales grew 9% and the two new locations reached breakeven
        a full quarter ahead of plan. The delivery pilot with regional
        couriers cut fulfilment cost per order by 18% and complaints about
        late deliveries dropped by half. Risks: the anchor mall of the
        Riverside store announced renovation for February–May, which
        historically cuts traffic by a third, and the regional head of sales
        resigned last week with a six-week handover.
        """,
    "online": """
        Online revenue grew 15%, but the number hides a split: subscription
        boxes grew 41% while one-off baskets shrank 6%. Checkout abandonment
        improved from 71% to 64% after the one-page redesign. Returns are the
        sore point — apparel return rate hit 34%, and processing cost per
        return rose to $6.10. The recommendation engine A/B added +2.3% to
        average order value and is ready to roll out fully. Ad spend
        efficiency worsened: blended CAC up 19% quarter over quarter.
        """,
}


class ReadReport(Tool):
    name        = "read_report"
    description = "Read one regional quarterly report in full."
    parameters  = {"type": "object",
                   "properties": {"region": {"type": "string",
                                             "enum": list(REPORTS)}},
                   "required": ["region"]}
    risk        = "read"

    def run(self, region, options=None):
        return textwrap.dedent(REPORTS.get(region, "no such report"))


# ── The scaffold ──────────────────────────────────────────────────────────────

agent = Agent(
    Model(MODEL, options={"reasoning": "none"}),        # workers: act, don't muse
    planner_model = Model(MODEL),                       # designer: think
    tools         = [ReadReport()],
    mode          = "waterfall",
    team          = "auto",
    instructions  = (
        "You coordinate a review crew. First design the team as your plan: "
        "which workers, with what remit each. Then delegate accordingly — "
        "your own conversation is for the plan and the final synthesis only."
    ),
    stop_when     = [step_count(12), cost_budget(0.50)],
)

result = agent.run(
    "Prepare an executive summary of the quarter across all three regions "
    "(north, south, online): top three wins, top three risks, and one "
    "decision the board must take."
)

# ── What happened ─────────────────────────────────────────────────────────────

print(result.output)
print("\n" + "═" * 72)
print("the plan (the team it designed):")
for i, s in enumerate(result.plan, 1):
    print(f"  {i}. {s['goal']}")

print(f"\nstopped_by : {result.stopped_by}")
print(f"steps      : {result.steps_taken}")
print(f"tokens     : {result.tokens_used:,}   ${result.cost or 0:.4f} "
      "(workers' spend included)")

print("\nJournal — who did what:")
for e in result.journal:
    obs = str(e.get("observation", ""))[:58].replace("\n", " ")
    print(f"  {e['seq']:>2}. [{e['outcome']:<6}] {e['intent']:<12} {obs}")

delegated = [e for e in result.journal if e.get("intent") == "delegate"]
print(f"\ndelegations: {len(delegated)}"
      + ("  — each result is the worker's own word (model_claim); the parent"
         "\n  did not verify it, and the journal says so instead of hiding it."
         if delegated else
         "  — the model chose not to delegate this time; the point of the"
         "\n  scaffold is that the composition is its choice, and visible."))
