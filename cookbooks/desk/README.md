# desk — a support agent that refunds small, asks for big, and moves to Lambda unchanged

Last run: aichain 4.10.1, 2026-10-17 — `run.py --check` and `simulate.py --check` 5 of 5, three runs out of three each.

## The case

An online shop's support inbox gets the same requests every day: *where is my
order*, *I want my money back*. The shop is happy for an agent to refund up to
$50 on its own, under the written refund policy. Anything above that a manager
signs off — and the manager may answer in five minutes or tomorrow morning.
Meanwhile the agent must not sit in memory waiting, and must not refund twice if
the process restarts.

This recipe builds that agent in one file, runs it in one process, and then runs
**the same agent, unchanged**, as two serverless handlers that share nothing but
a store: the first parks the conversation when approval is needed; the second,
triggered by the manager's answer, resumes it where it stopped. That is the
library's central promise — test locally, carry the same scenario to production
without rewriting it — and it is where the `agents` product begins.

*Needs: `ANTHROPIC_API_KEY`. No services: orders live in SQLite, parked runs in
files. A run of the four scenarios, measured: $0.046 and about a minute.*

## How it works

```
customer ─ lookup_order ─┬─ ≤ $50: refund                         (the policy approves)
                         ├─ > $50: refund ✗ "use refund_with_approval"
                         └─ refund_with_approval ─ parks ── … ── manager answers ─ resume ─ refund or not
```

- **The money rule is code, not a sentence in the prompt.** `refund` is a
  `@tool(risk="financial")`; `policy={"rules": {"financial": "approve"},
  "approve": up_to_50}` lets support refunds through up to $50 and refuses the
  rest with `ApprovalDecision(False, "over $50: use refund_with_approval")`. The
  agent reads the refusal and takes the other tool. A prompt that said "never
  refund more than $50" would be followed most of the time.
- **Waiting for a person is a park, not a sleep.** `gate(refund, …)` is a tool
  that pauses the run on its first call. `Agent(store=FileStore("data/runs"))`
  writes the conversation to a file and returns a `SuspendedResult` with a
  `run_id`; nothing stays in memory. `agent.resume(run_id, signal={"approved":
  …, "note": …})` — from any process, with an agent built afresh — runs the
  refund, or tells the model the manager's reason.
- **A retried refund refunds once.** The check and the write are one SQLite
  transaction, so a Lambda retry, or a model that calls twice, finds
  `refunded = 1` and stops.
- **One `Model` serves every tenant.** `Model(MODEL, api_key=tenant_key)`
  resolves the key per request from the ambient `RunContext`: the tenant's own
  `ANTHROPIC_API_KEY_<TENANT>` when there is one. The run document records the
  tenant's name; `--check` verifies no key reached `data/runs`.
- **A ceiling in money.** `max_cost=0.10` on the agent: the turn that would pass
  it does not begin.

### From here to production

`handlers.py` has Lambda's signature and imports nothing from AWS. What
`agents` adds around it: an event log in DynamoDB instead of the files in
`data/runs`, an orchestrator that receives the customer's message and the
manager's answer as events, an executor per turn, and the deploy. What does not
change is `desk.py`: `run.py`, `handlers.py` and `simulate.py` all call
`build_agent()`, and the difference between how they build the agent is zero
lines.

## Running it

```bash
pip install -r requirements.txt
python run.py --check            # four scenarios in one process, the manager answering in code
python simulate.py --check       # the same through handlers.py, each handler its own process
```

| Scenario | What happens |
|---|---|
| A-1007, $30, broken | refunded at once |
| A-1031, $300, broken | parked; the manager approves; refunded |
| A-1042, $120, broken | parked; the manager refuses, "outside the return window"; the reply says so |
| Where is A-1003? | answered from `lookup_order`; nothing written |

To stop between the two handlers and come back later — the manager answering
tomorrow — run them one at a time; the second process knows only the run id:

```bash
python simulate.py start big-refund-approved        # prints {"status": "parked", "run_id": "run-…"}
python simulate.py approve big-refund-approved run-…
```

`DESK_MODEL` or `--model` changes the model; each run starts from the same
fifty orders (`setup.py`).

## The original, side by side

The OpenAI Agents SDK's customer-service example
([`examples/customer_service/main.py`](https://github.com/openai/openai-agents-python/blob/a6ce52d255bf853c82df5ae19ff1482113e3128b/examples/customer_service/main.py)
at `a6ce52d`) is an airline desk: a triage agent hands off to an FAQ agent and a
seat-booking agent. Anthropic's
[customer-support-agent](https://github.com/anthropics/anthropic-quickstarts/tree/9bcc95e316e5ef6542b4c9d0469f4078829eead5/customer-support-agent)
quickstart (at `9bcc95e`) is a Next.js chat with answers from a Bedrock
knowledge base; it takes no actions, so there is no approval to compare.

| | OpenAI Agents SDK example | desk |
|---|---|---|
| Lines of orchestration | 148 | 64 (`desk.py` 47, `handlers.py` 17) |
| An action limited by amount | — | in the policy, in code |
| Waiting for a person | — | parks; resumes in another process |
| A retried action | — | refunds once |
| A key per tenant | — | `api_key=callable`, never stored |
| Cost of a conversation | not reported | per scenario, with a ceiling |
| Models | OpenAI | any; one word |
