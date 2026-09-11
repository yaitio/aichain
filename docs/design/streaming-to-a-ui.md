# Design — backing a conversational UI

Status: **shipped.** R1, R3 and R4 in `2.7.0`; R2 in `2.8.0`; R5's
library half in `2.6.0` (the gate) and `2.9.0` (the conversation). Written 2026-09-11
from the needs of one consumer: a conversational BI product whose front end
renders every tool result as a chart. (The header said "against `2.0.0`";
`Agent.stream()`, which R2 quotes, arrived in `2.5.0`.)

**What the code said when the requirements were checked against it.** Three
things the document assumed were not there. `run.suspended` does not exist —
it appears once, as an example in a docstring — so R5 had nothing to build on.
`run_id` was a declared field the agent never filled, so R3 was not "mostly
satisfied": of its three identity fields only `ts` was populated. And `approve`
was consulted and discarded, so the permission layer R5 extends was a policy
that gated nothing; that was fixed first, in `2.6.0`.

M3 shipped the step boundary — `Event`, `Hook`, `Tracer`, permissions. This
document says what is still missing before that boundary can drive a user
interface rather than a log file, and it deliberately stops at the library's
edge: transport, HTTP and wire formats stay in the application, exactly as
`_events.py` already says of tracing sinks.

## The consumer, in one paragraph

The product holds a conversation over MCP sources and draws the answer. A chart
is chosen from the **structure of the tool result** — column count, measure
type, object count — never from the model's prose. Pinning a chart to a
dashboard stores the *call*, not the data, so the dashboard refreshes without
the model. An answer is worth something only if it can be reproduced from a log
of its calls.

Three consequences follow, and they are the whole of this document:

1. the raw tool result must leave the loop while the run is still going;
2. each call must be identifiable, because several run in one turn;
3. the final prose must arrive incrementally, because a reader watches it.

## Sequencing — what blocks a migration and what does not

The consumer ships its MVP on another framework (the Vercel AI SDK) and migrates
here when this document is satisfied. That makes the order concrete rather than
a matter of taste:

| | blocks the migration | why |
|---|---|---|
| **R1** — result on the event | **yes** | without it there is nothing to draw; nothing downstream can recover what a prose rendering discarded |
| **R2** — streamed answer | **yes** | an answer that appears all at once after a long silence is a regression the consumer cannot ship |
| **R3** — identity and order | yes, but R1 closes it | needs only the id R1 already adds |
| **R4** — refusal against emptiness | yes, but R1 closes it | falls out of carrying the result verbatim |
| **R5** — approval on the channel | no | the consumer has no human-in-the-loop step at MVP; wanted, not blocking |

So the blocking set is **R1 and R2**, and R1 is the larger of the two because it
changes what an `Event` is. Everything else either follows from R1 or can land
later without holding anyone up.

## The governing constraint

> **The event channel must carry enough to reconstruct the turn — not enough to
> describe it.**

Today it carries a description. `step.started` emits `payload={"tool":
call.name}`; `step.ended` adds `error`. The call's arguments and its result
reach only two places: the message list, which is the model's view, and the
journal, where the result has already been rendered to
`prompts.observation_text(...)` — prose for a model, not data for a program.
The comment there is explicit that media is named rather than embedded.

That is the right decision for a journal. It is the wrong decision for the only
channel a UI can watch.

---

## R1 — Tool events carry the call id, the arguments and the raw result

**Required.** `ToolCall` is already `{id, name, arguments}` and `call.id`
already survives into `tool_result_turn`. The data exists; it does not reach the
event.

```python
self._emit("tool_call.started",
           name=call.name,
           payload={"id": call.id, "arguments": call.arguments})
result, error = self._execute(call, state)
self._emit("tool_call.ended",
           name=call.name,
           payload={"id": call.id, "result": result},
           error=error)
```

The names `tool_call.started` / `tool_call.ended` are already documented in
`_events.py` and in `Hook`'s dispatch rule; the agent emits `step.*` instead.
Either the docs or the emission should move, and the docs are the better guide:
a step and a tool call are not the same thing in the plan-driven modes.

**Why the result and not a rendering of it.** A consumer that draws the result
needs the source's own structure — units, additivity, column metadata, the
`notes` a source attaches, and the difference between *no rows* and *the tool
declined*. Every one of those is lost by a text rendering, and none of them can
be recovered downstream.

**This changes what an `Event` is**, and that is a decision for the library, not
for a consumer. The docstring says an event carries "only operational fields".
A tool result is data. Two honest readings:

- *by value* — `payload["result"]` holds it. Simple; events stop being small.
  Measured on the consumer: a 1 000-row result is 92 143 bytes uncompressed.
- *by reference* — the event carries a handle, the application fetches it.
  Events stay small; the application gains a lifetime to manage, and a
  streaming consumer pays a round trip per call.

**Settled: by value.** The argument the document makes is right and there is a
stronger one it does not make. A handle needs a lifetime and somewhere to live
— that is state between invocations, and the library's niche is Lambda, where
`VISION.md` names synchronicity and connection-per-call as advantages not to
be "fixed" into a daemon. The process that issued the handle may be gone. So
events stop being uniformly small, and `Event.__repr__` omits the payload so a
log stays readable.

**Acceptance.** A `Tracer` attached to a run over a tool returning a structured
document can rebuild, for every call: which tool, with which arguments, what
came back, in what order, and whether it failed — without reading the message
list and without parsing prose.

---

## R2 — The agent streams the answer, not only its actions

**Required.** `Agent.stream()` yields events and says why:

> A turn in an agent loop is usually a tool call, not prose, so token deltas
> would be empty for most of a run.

True of the middle of a run, false of its end. The last turn is the answer, a
reader is watching it, and today it arrives in one piece after a silence as long
as the model takes. `Skill.stream()` already produces text deltas; the agent
does not route them.

Required: text deltas reach the same channel as the events, tagged with a
message identity so that a consumer can open, append to, and close one block of
prose. Whether that is a new event type (`text.delta`) or a second channel is
the library's call; a single ordered stream is easier to consume.

**Acceptance.** A consumer driving `Agent.stream()` can render the final answer
incrementally, interleaved in the correct order with the tool calls that
preceded it, without a second call to the model.

---

## R3 — Identity and order are sufficient to reconstruct the turn

**Required, mostly satisfied.** `Event` carries `run_id`, `step`, `ts`,
`duration`, `usage`, `cost`. With R1's `id` added, a consumer can attribute
concurrent calls correctly.

The remaining gap is **parallel calls in one turn**. The agent already honours
every call a provider requests — deliberately, and the comment explains why —
but without an id per event, a consumer watching two calls in flight cannot tell
which result belongs to which. R1 closes this; R3 only states the requirement
that it must stay closed, including if calls later execute concurrently rather
than in sequence.

**Acceptance.** Given a turn with three tool calls, a consumer can pair each
result with its arguments using event fields alone.

---

## R4 — A refusal is distinguishable from an empty result

**Required.** Three outcomes must not collapse into one:

| outcome | today | needed |
|---|---|---|
| the tool raised | `error` on the event | kept |
| the tool answered, zero rows | indistinguishable from below | the result, verbatim (R1) |
| the tool declined, and said why | indistinguishable from above | the result, verbatim (R1) |

R1 delivers this as a side effect, because a source that declines says so inside
its own response. R4 exists to name it, because a consumer that cannot tell a
refusal from an empty answer will draw an empty chart for both, and an empty
chart is a lie about the data.

**Acceptance.** A tool whose protocol expresses refusal in its return value
produces an event from which a consumer can tell refusal from emptiness without
string-matching prose.

---

## R5 — Approval travels on the same channel

**Required for human-in-the-loop, deferrable.** `tools/_permissions.py` has risk
classes and `ALLOW` / `APPROVE` / `DENY`; `run.suspended` exists. A UI that must
ask a human needs the request and the answer to be events like any other, so
that one consumer sees one ordered stream:

- a request naming the call id, the tool and the risk class;
- a response carrying the decision and, when denied, the reason;
- a terminal event for a call that was denied, so the consumer stops waiting.

**Acceptance.** A UI can present an approval prompt, return a decision, and
render the outcome, with no access to the permission layer beyond the event
stream.

---

## Explicitly not required of the library

**HTTP, SSE and any wire format.** `_events.py` already draws this line for
tracing sinks — "OpenTelemetry/file/DB sinks live in the application" — and the
same line holds here. Encoding events as `data: {...}` frames, choosing status
codes, and setting anti-buffering headers are the application's job.

**Vercel's UI Message Stream protocol.** It would buy exactly one thing:
`assistant-ui` reads it without an adapter. It costs tracking a format the
library does not control — within `ai@7.0.47`,
`toUIMessageStreamResponse` is already deprecated in favour of two newer
helpers. An application that wants that protocol can map R1–R5 onto it in one
module; the library should not carry it.

**Persisting the journal.** A durable log is the product's obligation, not the
library's. R1–R3 make it possible; storing it is downstream.

**A different agent loop.** Nothing here asks the agent to behave differently.
Every requirement is about what leaves it.

---

## Open questions — two settled, one still open

**Result by value or by reference (R1). Settled: by value**, for the reason
above: a handle is state between invocations and the target is Lambda.

**`step.*` against `tool_call.*` (R1). Settled: renamed.** The deciding
argument was not the disagreement with the docs but one the document does not
mention: **`Chain` emits `step.*` too**, for a chain step, so a hook attached
to a chain containing an agent received both under one name and could separate
them only by the shape of the payload. That does not get fixed by
documentation. A `Hook` subclass with `step_started` / `step_ended` still
fires, once, with a `DeprecationWarning` — the rename is loud rather than
silent, and costs nobody a run.

**Where text deltas live (R2). Settled: one ordered stream**, as recommended —
`Agent.stream()` was already a view of the hook channel rather than a
vocabulary beside it, so `text.started` / `text.delta` / `text.ended` landed
there without a new mechanism.

What the requirement did not foresee is what streaming the agent's turns
**costs**: a stream cannot use the fallback chain, and it cannot retry once a
piece is out — while the loop retries every model call deliberately, because
one that does not pays for each transient blip with a lost decision. So
`run()` stays buffered (it has nobody to show pieces to) and the retry window
was reopened for streams: transient failures retry while nothing has been
yielded. The 2.5.0 note refusing retries outright, on the grounds that such a
rule "holds sometimes", was wrong — before the first piece is a state the
caller can see.

**R5 split, and the split held.** The mechanism — `approve` actually blocking
— shipped in `2.6.0` as `Agent(approve=...)`, fail-closed when nobody is there
to ask. The *conversation* shipped in `2.9.0` as `approval.requested` /
`approval.decided`, plus `ApprovalDecision(False, reason)` so a refusal can say
why.

**It did not reopen the `2.0.0` decision after all**, which is worth recording
because the opposite was expected. Suspend/resume would only be needed if the
library owned the *wait*; it does not. The approver is an ordinary callable,
so a UI's answer arrives by whatever means the application already has, and
the library carries the question and the answer without parking a run. Across
processes the wait is the product's, exactly as this document says of
transport.

The requirement that had to be **built** rather than described was the
ordering. The gate lived inside the tool call, so its events were drained at
the next boundary — after `tool_call.ended` — and a prompt delivered after the
decision is a record, not a prompt. Moving the gate to the loop boundary is
also better placed: a permission is a property of the turn, not of the
invocation.

---

## Why this fits the library's own filter

VISION's test for a feature is whether it moves an axis from *hardwired* to
*injectable*. What is hardwired today is the **consumer of a tool result**: it
goes to the model, and to a journal that renders it for the model. Nothing else
can see it while the run is alive. R1 makes that audience injectable — the model
keeps its view, and a program gets one too.

R2 does the same for prose: today the only consumer that can watch an answer
being written is `Skill`; the agent's answer has exactly one audience, arriving
at the end. R5 does it for approval, which today can only be a policy and never
a conversation.

None of the five asks the library to know what a chart is, what MCP is, or what
the product does with any of it.
