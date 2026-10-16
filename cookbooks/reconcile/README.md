# reconcile

Last run: aichain 4.10.1, 2026-10-17 — every strategy solved 3 of 3 default cases on `claude-haiku-5-5`, $0.003 a case for the agents and $0.034 for the scaffold.

Check a supplier's invoices against the contract and the delivery notes, and
report every discrepancy — with four agent strategies on the same cases, and an
answer key that costs nothing to check.

*One key, `ANTHROPIC_API_KEY`. No services, no files: the cases are generated.*

## The case

Reconciliation is a job every company already has — accounts payable,
procurement, logistics — and it is the kind of back-office work that pays for
an agent. A case is a contract in two parts, three invoices and their delivery
notes:

```
INVOICE INV-2026-1000  period 2026-04-16..2026-05-15  EUR
    #  code       qty     price     total
    1  CRT-SML      4     11.25     45.00
    2  CRT-LRG      2     19.80     39.60
    3  DLV-URB      3     95.00    285.00
       TOTAL                     451.30
```

The faults are planted by the generator, so the answer key is exact: no judge
model, no hand labelling, and a sweep of twenty cases is scored by arithmetic.
That is why this project is also **the instrument other recipes here are
checked with** — any claim that a strategy reconciles better can be put to a
number for the price of the calls.

They sit on three depths, and the ladder is what lets the cases tell strategies
apart:

| depth | needs | kinds |
|---|---|---|
| **local** | one document | bad line arithmetic, wrong invoice total, a charge billed twice, wrong currency |
| **cross** | a second document | price against the contracted rate, quantity against the delivery note, period against the term |
| **derived** | a conditional clause plus date arithmetic | a delivery late under clause 7.2 with no penalty credit applied |

## How it works

Every strategy answers through the same three tools, so **the actions are the
answer** and no output format is parsed:

| tool | |
|---|---|
| `list_documents` | what is in this case |
| `read_document` | one document, in full |
| `report_finding` | one fault: kind, document, line, and the two values that disagree |

A discrepancy described in prose and never reported does not count — an agent
that narrates instead of acting has not done the job.

The four strategies:

```python
Agent(model, tools)                     # agile — the loop decides each step
Agent(model, tools, mode="waterfall")   # a plan written at step 0, then held
Agent(model, tools, team={"extend": True})   # hires workers when it wants them
run_scaffold(case, model)               # no Agent: Model + Skill + Pool
```

The first three are one `Agent` with one setting changed, which is the point:
the comparison is between strategies, not between harnesses. The fourth never
constructs an `Agent`. A coordinator reads the case *and a price list*, decides
how many reviewers to buy and what each looks for, a `Pool` runs them at once,
and a consolidator merges what they found. It competes for the same job without
fitting the same interface, so it is here as a control — what hand-built
orchestration costs next to the loop.

Every strategy gets the same leash — `step_count(30)`, `token_budget(150_000)`,
`max_cost=0.60` and an 8 000-token output cap; the scaffold shares one `Budget`
across its coordinator, its pool of reviewers and its consolidator. Every model
call is counted on the strategy's own receipt (`receipt.py`), from the events
the library sends, so the comparison is in dollars. A strategy left at a
different ceiling is being compared at a different price.

**Scoring is one bit per case.** `solved` means every planted fault found and
nothing invented, because recall alone is gameable: flag every line and it
reaches 1.0. `found` and `false` sit beside it so partial credit and invention
stay visible. A line-level finding matches on location — document and line —
not on its label; a document-level one has no line, so there the label is the
location. Recall, precision, label agreement, cost per fault and recall by
depth are diagnosis, behind `--detail`, and pooled across cases with the counts
printed beside every ratio.

## Running it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...

python generate.py --check                        # 200 cases, the oracle checked against itself
python reconcile.py --cases 3                     # all four strategies, claude-sonnet-5-5
python reconcile.py --cases 3 --model claude-haiku-5-5 --modes team,scaffold
python reconcile.py --cases 10 --defects 8 --invoices 5 --detail
```

The default — three invoices, four faults — is easy for current models; every
strategy may solve it. **Turn the dial** before reading anything into a
ranking: more defects and more invoices push the cross and derived faults out
of a single read. Faults are drawn without replacement, so eight defects are
eight different kinds.

The scaffold's coordinator chooses from the `ROSTER` at the top of
`reconcile.py`; it needs only `ANTHROPIC_API_KEY` as shipped, and any model
the library reaches can be added to it — the prices beside each are what the
coordinator reads when it chooses. Results swing between runs on a single
case: three cases is a smoke test, ten is the least a claim should rest on.

Nothing here reproduces an existing project, so there is no side-by-side.

While it was being built, this benchmark found four defects in the library,
each since fixed in aichain.
