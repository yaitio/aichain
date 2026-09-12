# Changelog

## [Unreleased]

## [2.11.0] — 2026-09-12

**Judging.** Both items carried over from `2.3.0`, and the reason they matter
is that the first thing needing them cannot be built from what was here: every
number in the byheart study is recall of evidence, and nothing scores the
answer.

### Added

- **`pairwise(model)` — a comparison that is not measuring position.** LLM
  judges have a documented preference for whichever answer they read first, so
  a single-ordering comparison measures that preference alongside quality and
  produces a number that looks exactly like a quality score. The pair is put
  twice, A/B and B/A, and the candidate wins only by winning both.

  A disagreement between the orderings **is** the position bias showing
  itself, and it resolves to the champion: the burden is on the challenger,
  which is what keeps a best-so-far from drifting on noise. How often they
  disagreed is reported per verdict — that is the judge's own reliability,
  measured for free while grading, and a pair that disagrees half the time is
  a coin toss wearing a rubric.

- **`abstain(why)` — a verdict of no verdict.** `ok=False`, so nothing that
  ignores the flag can read it as a pass, and excluded from the denominator,
  so nothing reads it as a wrong answer. Those are different claims: "wrong"
  and "unreadable by the instrument" fold into one number only if you are
  willing to blame the arm for the judge.

### Changed

- **`judge()` abstains instead of raising** on an unparseable verdict. Raising
  was right about the important half — guessing "pass" raises every arm at
  once and reads as a good result — and wrong about the rest: it ended the
  whole run over one row a judge could not read.

- **`Report` counts and prints the exclusions** (`abstentions()`, an `n/j`
  column). A denominator that quietly shrinks is its own defect. And an arm
  the judge could not read for more than a fifth of its rows is **rejected**:
  excluding rows keeps the survivors honest, but past that share there are no
  survivors worth printing — 0.95 of the fifth it managed to read is not a
  result.


## [2.10.0] — 2026-09-12

**A tool call with no result never reaches a provider.** The history invariant
on the externally driven seam, outstanding in `PLAN.md` since the τ²-bench
study.

### Changed

- **`step()` refuses a malformed history**, naming the unanswered call ids and
  what to append. Its docstring already handed the obligation to the caller —
  *"the caller appends the reply and whatever the world answered, then calls
  again"* — and nothing verified that they did. That seam is what a benchmark
  harness and a serverless driver use, which is exactly where a result crosses
  a process boundary and can be lost.

  It matters more than it looks because the providers disagree in the worst
  possible way. OpenAI chat completions, the Responses API and Anthropic all
  reject the request; Google keys results by name and is looser; a self-hosted
  OpenAI-compatible server — vLLM, llama.cpp, mlx — validates **nothing** and
  templates whatever it was handed, so the model meets its own unanswered call
  and improvises: repeats it, apologises, or invents the result, differently
  each time. Provider interchangeability is this library's main promise and
  this is a place it did not hold.

  And the condition that produces a dangling call — a tool that raised, a
  result dropped across an invocation boundary — is itself intermittent, so it
  became unbounded behaviour on some trials and not others: run-to-run
  variance manufactured by us. The check does not turn a wrong answer into a
  right one; it turns quiet weirdness into the same named failure every time,
  locally, before the network, pointing at the caller rather than at a
  provider's wording. An instrument for reliability, not for accuracy — and on
  a benchmark a crash can rank below a model muddling through, which is worth
  knowing before reading the next score.

  `run()` is unaffected: it appends a result turn for every call in a reply,
  failed ones included, and always did. Turning the check on cost the existing
  suite nothing, which is the evidence for that.

### Added

- **`dangling_calls(messages)`** in `models._calls` — the invariant as a
  function, so a driver can ask before it commits rather than learn from the
  raise.

### Fixed

- **The documented external-driver example could not run.** It read
  `agent.execute(decision["action"], state)`; `step()` returns a
  `ToolCallRequest` or a string, never a dict. The paragraph after it
  described reflection assigning a `store_as` key into memory — gone since
  `2.0.0` — and the modes table still listed a third mode, `goal`. Replaced
  with the whole loop, including the appending the invariant now requires.


## [2.9.3] — 2026-09-12

**A tool's schema becomes authority over its options.** They already declared
them and nothing read them: `searchPerplexity` describes seven option keys,
and `options={"recencyy": "day"}` passed validation, reached the tool and was
dropped on the floor — the search ran unfiltered and answered confidently. A
typo cost a run and left no trace. The same shape as the model options fixed
in 2.3.0, one layer down.

### Added

- **`Tool.unknown_keys()`**, and both call paths use it:

  * a **model's** call gets a remediation message from `check_args` naming the
    wrong key and the accepted ones, which an Agent feeds back inside the
    step's attempt budget;
  * a **person's** call gets a warning from `_validate`, not a refusal. The
    asymmetry is deliberate: a subclass may legitimately read an option its
    schema does not advertise, and refusing someone's call over the library's
    reading of their own schema would be the library knowing better.

  One level into a declared object and no further. The top level already fails
  loudly — an unexpected keyword reaches `run(**kwargs)` and Python raises —
  and a nested object with no `properties` of its own is a free-form dict by
  declaration, left alone. A schema is authority only over what it describes.

  It reaches most of the surface: 28 of 32 tool classes describe their
  options, and the number is now asserted so a drop is visible.

### Fixed

- **`ttsQwen` read an option it did not declare.** The other direction, and
  the worse one: `region` is honoured by the tool and absent from the schema,
  so a **model could not discover it at all**, and a caller who used it was
  passing a key the schema calls unknown. Declared on `ttsQwen` alone —
  putting it on the shared TTS schema would advertise it on OpenAI, Google
  and xAI, where nothing reads it, which is the opposite defect and the one
  2.3.0 was about. A test now checks every tool for the same disagreement;
  this was the only one.


## [2.9.2] — 2026-09-12

Two silent divergences in what the providers were actually sent, both in the
handling of the system prompt, and both invisible to every example this
repository ships.

### Fixed

- **The Responses API path kept only the last system message.** `instructions`
  was **assigned** inside the loop over messages rather than accumulated, so a
  second system message replaced the first. Present since the very first
  commit (`0211386`, 1.0.0) and carried through the 1.2.3 migration verbatim.

  Scope, measured rather than estimated: `gpt-5.4` / `5.5` / `5.6` only, and
  only when the caller wrote two or more `system` messages — one is the common
  shape and always worked, and `Agent` emits exactly one. Nothing shipped in
  this repository could reach it: of 24 examples one carries a system prompt
  (a single one, on Anthropic) and one runs a Responses model (through an
  Agent). The prompt did not fail, it arrived **partially** — which is worse
  than failing, because the answer is plausible and the model gets the blame.

  The plan had this recorded as *"Responses overwrites, the rest
  concatenate"* — as a **property of the provider**. It was ours: the
  `instructions` field is one string, so joining several system messages is
  the library's job, and every other family does it. A defect filed as a
  provider difference is one nobody fixes, because provider differences are
  supposed to differ.

- **A mid-conversation system message meant two different things.** Anthropic,
  Google and the Responses API each carry the system prompt in a field of
  their own, so one written in the middle is hoisted out of the sequence and
  applies from the start; the OpenAI-compatible family left it in place, where
  it reads as an instruction that begins at that point. The same conversation,
  two meanings, nothing said.

  Three providers cannot express the positional reading at all, so global is
  the only portable one, and it is now global everywhere — hoisted on the
  OpenAI family too, and **reported** as an adaptation, because for that
  family it is a real change to what the model sees.

### Added

- **The conversations no example contains**, compared across all nine text
  providers: a system message in the middle, system messages at both ends, an
  assistant turn first and in the middle, two user turns in a row, multi-part
  messages, empty parts. What is compared is the **meaning** — is the system
  prompt global, what is the sequence of turns — not the wire, because the
  wire legitimately differs: Google says `model` where the others say
  `assistant`, and three providers carry the system prompt in a field.

  Both defects above lived in a shape our own corpus cannot produce. The
  examples had been doubling as the test set, and the test set never contained
  the case.


## [2.9.1] — 2026-09-12

The five instruments `PLAN.md` has been asking for since the June audit, and
the defect the third one found on its first run.

### Fixed

- **`format: {"type": "json"}` on Anthropic did nothing and said nothing.**
  The request went out **byte-identical** to a plain-text one: the caller
  asked for JSON, got prose, and `parse_response` was left digging an object
  out of whatever came back. Anthropic has no JSON-mode field and every other
  provider does, so by the universal-vs-local rule it is absorbed rather than
  refused — carried as a system instruction and reported as `adapted`.
  (`json_schema` was never affected: it is a forced tool call and works.)

  Silence about a format is worse than silence about a sampling knob. A
  dropped `temperature` gives a different answer; a dropped JSON mode gives
  the wrong *type*.

### Added — tests

- **A lightness-invariant test.** `VISION.md` calls the README hello world an
  architectural guarantee — no new mandatory parameter, ever — and says a
  `Skill.run()` that got more complex is the signal that the environment
  leaked into the scenario. Nothing watched for that signal, and the README's
  *agent* example meanwhile carried a `TypeError` on its first line through
  four minor versions. The hello world now runs against a fake transport with
  exactly the arguments the page shows, and every other fenced example is
  **bound** against the real signatures — required arguments supplied, arity
  right, which is the half the existing docs test could not see.

- **Lifecycle smokes** — create → save → load → run, for every primitive that
  serialises. Each individual piece of the round trip worked when it was
  broken last week; only running it in one breath showed that what came back
  was not what went in. Where a primitive deliberately does not serialise,
  that is asserted rather than skipped: a missing test and a deliberate
  absence look identical in a run, and only one is a defect.

- **One input across every text provider.** The instrument for the class where
  a request works on six providers and quietly does something else on the
  seventh. It asserts what must be true everywhere — the system prompt
  travels, JSON is asked for, the answer has a ceiling — and separately pins
  where they *legitimately* differ, so a check that had to be loosened until
  it passed everywhere would be visible as such.

- **The chunker's contract**, which is arithmetic and therefore the one part a
  reader cannot verify by looking at output and finding it plausible. Every
  assertion is a sentence already in the module docstring.

### Changed — tests

- **The legacy import bridge is gone.** `conftest.py` aliased every
  sub-package into `sys.modules` under its pre-2.0 top-level name; written as
  temporary, it lived three months and cost more than it looked. `tests/agent/`
  is itself a package named `agent`, so inside that directory the alias lost
  to the real one and an import resolved differently depending on where the
  file sat; a module imported under two names is two module objects with two
  copies of every module-level value; and nothing ever failed, so the
  migration it existed to enable never happened. All 37 files import
  `yait_aichain.*` now — the suite reaches the library the way its users do.

  Two things the mechanical rewrite got wrong, both caught: `patch()` takes
  its target as a **string**, so those were left behind (they raise, which is
  the one mercy), and one search string inside a docs test was rewritten into
  a module path, leaving a test that passed while looking for something no
  page could contain. A blind rewrite is exactly as blind as `getattr` with a
  default.


## [2.9.0] — 2026-09-11

**Approval travels on the same channel.** R5 of
[design/streaming-to-a-ui.md](docs/design/streaming-to-a-ui.md), and the last
of the five. A UI can now present the prompt, return a decision and render the
outcome without reaching into the permission layer at all.

### Added

- **`approval.requested` · `approval.decided`**, carrying the call id, the
  tool, the risk class, the arguments, and then the verdict with its reason.
  Emitted **even when there is nobody to ask**: otherwise the call simply
  fails and nothing says it was a governance decision rather than a broken
  tool.

- **`ApprovalDecision(granted, reason)`** — return it instead of a bare
  `False` and the refusal says why. The reason travels into the event and
  into the denial the model is told about. "Not approved" and nothing else
  throws away the only part a person can act on, and it cannot be recovered
  afterwards: it was in the head of whoever clicked no.

- `ApprovalRequest` and `ApprovalDecision` are importable. The approver has
  been handed an `ApprovalRequest` since `2.6.0` and could not import the
  type to check against it.

### Changed

- **The permission gate moved from the tool call to the loop boundary.**
  This is the requirement that had to be built rather than described. Inside
  the call, the gate's events were drained at the next boundary — after
  `tool_call.ended` — so a streaming consumer received `approval.requested`
  only once the decision had been made. The record read correctly and was
  useless for the one thing R5 exists for. Now the request reaches the
  consumer **before** the approver is called, which is also the better place
  for it: a permission is a property of the turn, not of the invocation.

  The human is asked once — a second gate left inside the call would put the
  question to a person twice.

### Notes

**R5 did not reopen the `2.0.0` decision**, which is worth recording because
the opposite was expected when the requirements were reviewed.
Suspend/resume would be needed only if the library owned the *wait*; it does
not. The approver is an ordinary callable, so a UI's answer arrives by
whatever means the application already has, and the library carries the
question and the answer without parking a run. Across processes the wait is
the product's half, exactly as the document says of transport.

### Fixed

- **The second test double only answered the buffered call**, the same defect
  fixed in the agent's double in `2.8.0` — and it mattered here, because the
  tests that needed it are about streaming: the fake would have failed the
  feature rather than the code. Both doubles stand in for a provider in both
  of its modes now.


## [2.8.0] — 2026-09-11

**The agent streams the answer, not only its actions.** R2 of
[design/streaming-to-a-ui.md](docs/design/streaming-to-a-ui.md).
`Agent.stream()` said why it did not: *a turn in an agent loop is usually a
tool call, not prose, so token deltas would be empty for most of a run*. True
of the middle of a run and false of its end — the last turn **is** the answer,
a reader is watching it, and it arrived in one piece after a silence as long
as the model takes.

### Added

- **`text.started` · `text.delta` · `text.ended`** on the same channel as
  everything else, all carrying one `payload["id"]` so a consumer can open a
  block of prose, append to it and close it. One ordered stream rather than
  two to reassemble, and interleaved correctly with the tool calls that
  preceded it. A turn that asks for a tool and says nothing produces none of
  the three: an empty open/close pair is something a consumer would have to
  filter.

  `run()` emits none of them. It has nobody to show pieces to, and buffering
  keeps the fallback chain.

### Changed

- **A stream retries while the attempt is still discardable.** `2.5.0`
  refused to retry at all, reasoning that a rule holding only before the
  first byte "holds sometimes". That was wrong — *before the first piece* is
  not a sometimes, it is a state the caller can see, because nothing has been
  yielded — and the agent is what made it matter: its loop retries every model
  call deliberately, since one that does not pays for each transient blip with
  a lost decision. Transient failures (429, 5xx, network) now retry until the
  first piece is out and are raised after it.

  The fallback chain stays out of a stream on purpose: a second model is a
  different answer, not a retry of this one, and swapping models mid-sentence
  is not a thing to do quietly. That cost is why `run()` does not stream.

### Fixed

- **The agent test double only answered the buffered call.** It mocked
  `_post` and not `_post_sse`, so the moment the agent streamed, every tool
  event vanished and the run ended in a network error — which reads as the
  feature being broken rather than the fake being half-written. A double that
  stands in for a provider has to stand in for both of its modes.


## [2.7.0] — 2026-09-11

**The event channel carries enough to reconstruct a turn.** R1, R3 and R4 of
[design/streaming-to-a-ui.md](docs/design/streaming-to-a-ui.md). It carried a
description: `step.started` emitted the tool's name and nothing else, and a
call's arguments and its result reached only the message list — the model's
view — and the journal, where the result had already been rendered to prose
for the model, media reduced to `[returned 2 image]`. That is right for a
journal and wrong for the only channel a program can watch while a run is
alive.

### Changed

- **`step.*` → `tool_call.*` on the agent.** The documented names were never
  the emitted ones, but the deciding argument is a different one: **`Chain`
  emits `step.*` too**, for a chain step, so a hook attached to a chain
  containing an agent received both under one name and could tell them apart
  only by the shape of the payload. A `Hook` subclass with `step_started` /
  `step_ended` still fires, once, with a `DeprecationWarning` naming the new
  method — the rename is loud, and costs nobody a run.

- **`tool_call.*` carries the call.** `payload["id"]` (a model may ask for
  several in one turn and the agent honours all of them, so the id is how a
  result pairs with its arguments), `payload["arguments"]` on started, and the
  tool's **raw** result on ended.

  By value, and that is a decision about what an `Event` is. A rendering
  cannot be recovered downstream — units, column metadata, and the difference
  between *no rows* and *the tool declined* are gone once it is prose, and a
  consumer that cannot tell a refusal from an empty answer draws an empty
  chart for both. A handle instead would need a lifetime and somewhere to
  live, which is hostile to the serverless target: the process that issued it
  may be gone. Events are therefore no longer uniformly small — about 90 KB
  for a thousand-row result — and `Event.__repr__` omits the payload so a log
  stays readable.

### Fixed

- **`run_id` was a declared field the agent never filled.** Two concurrent
  invocations — the ordinary case for the serverless target — wrote into one
  stream that could not be demultiplexed afterwards. Every event of a run now
  carries it, **including the `llm_call.*` a `Skill` emits inside the run**: a
  Skill knows nothing of the run it is inside, so the agent stamps identity on
  what its children emit rather than asking them to know.

- **`step` on a tool event**, so a consumer can say which turn a call belongs
  to.

### Notes

R2 (text deltas from the agent's last turn) and R5 (approval as events) stay
open. R5's *mechanism* shipped in `2.6.0` — `approve` gates now — and its
*conversation* is the product's half; putting the request and the response on
the event stream reopens the deliberate `2.0.0` decision that the agent has no
suspend/resume, which should be said out loud rather than arrive inside a UI
feature.


## [2.6.1] — 2026-09-11

The documentation sweep 2.6.0 ratcheted, and the code defect it uncovered on
the way: **a chain containing an agent could not be saved and reloaded at
all.**

### Fixed

- **`Chain.save()` lost the agent and `Chain.load()` then crashed.** The
  serialiser read the agent through `getattr(runner, "...", default)` under
  the pre-`2.0.0` names — `orchestrator`, `max_steps`, `max_attempts`,
  `max_tokens`, `persona`. Every one had been renamed or removed, so every
  read fell back to its default: the file lost the model, the instructions,
  the stop conditions and the labels, and gained three invented budget
  numbers. Load then raised `TypeError` on the null model name.

  The defaults are what made it quiet, and the lesson generalises: `getattr`
  with a fallback turns a renamed attribute into a plausible value, and
  plausible values do not fail tests. This shipped through four minor
  versions with a documented "Persistence inside `Chain.save()`" section
  describing it.

  A ceiling now carries a `spec` naming itself and its number, so
  `step_count` / `token_budget` / `cost_budget` survive the trip. A `check`
  cannot — its predicate is the caller's own function — and **saving one
  warns** rather than dropping it silently. A file written by the broken
  versions is refused on load with what to do, instead of being rebuilt into
  a different agent.

### Fixed — documentation

**34 documented parameters that do not exist, down to 0.** All of them the
agent, all of them 2.0 renames that never reached the pages teaching the API:

| documented | actually |
|---|---|
| `orchestrator` (13 pages, README included) | `model` |
| `persona` | `instructions` |
| `max_steps`, `max_tokens`, `max_attempts`, `done_when` | entries in `stop_when` |
| `memory`, `store`, `executors` | gone |
| `allow_spawn` | `team` |
| `mode="goal"` | `"agile"` already is that |

Three pages were rewritten rather than patched, because each taught a
subsystem that is gone: `configuration.md` (the whole pre-2.0 parameter set),
`memory.md` (the agent has no memory — the word does not occur in its source,
and `AgentResult.memory` is never written to), and the agent halves of
`state.md` and `observability.md`, which documented crash recovery and a
cross-process serverless pattern through `Agent(store=...)` and
`agent.resume()` — removed in `2.0.0` when the agent's state became the
conversation.

Two further claims were false and are now stated plainly: a `Gate` tool inside
an agent does not pause it (the `Suspend` it raises is caught like any tool
failure and reported to the model as an error), and the two documented
automatic stop rules — "no progress" and repetition — are fired by **no
library code**: `Journal.has_progress()` and `Journal.is_repeating()` exist
and are called from nowhere.

Every decision behind every rename is recorded, in full and with reasons, in
`docs/design/default-agent.md`. The design record was kept; the pages were
not touched. That is the same failure as the provider defaults losing their
authorship in 1.2.3 and as the changelog living outside git — the reasoning
survives and does not reach the person who needs it.


## [2.6.0] — 2026-09-11

**A permission policy that actually permits.** `approve` was the decision the
shipped defaults give to `external`, `financial`, `privileged` and to any risk
class nobody classified — and the agent consulted the policy for `deny` and for
nothing else, so a gated tool ran. A policy attached in order to gate spending
gated nothing while reading as protection in every docstring.

### Changed

- **`approve` now gates.** The Agent asks its new `approve=` callable and runs
  the tool only on a yes. The callable is handed an `ApprovalRequest` carrying
  the tool's name, its risk class, **the arguments it would run with**, the
  call id and the asking agent's name — approving a name rather than a call is
  approving nothing, since the arguments are what separate a $5 refund from a
  $50,000 one.

  **With no approver attached the call is refused.** A decision whose entire
  content is "a human should see this first" cannot resolve to "go ahead"
  because no human was configured. This is breaking for anyone who attached a
  policy and relied on it doing nothing; the refusal names both ways out —
  attach an approver, or set that risk class to `allow`.

  A refusal comes back through the tool channel as a result, not as a crash:
  the model is told and can choose something else. A denied call it never
  hears about is one it will simply make again. A delegated worker inherits
  the approver along with the policy.

### Fixed — documentation that taught an API that is gone

The module's own docstring promised `approve` would "pause the run for an
external approval, reusing suspend/resume", and two pages showed the flow with
`Agent(store=...)` and `agent.resume(...)`. `2.0.0` removed the agent's
suspend/resume deliberately — its state **is** the conversation — so those
examples could not have run either. Corrected, including the cross-process
serverless pattern, which was written with an `Agent` on both sides and is a
`Chain` capability.

A `Gate` tool handed to an agent does not pause it: the `Suspend` it raises is
caught like any other tool failure and reported to the model as an error. Said
out loud now rather than implied.

### Added

- **`tests/test_docs_promise_what_exists.py`** parses every fenced example and
  checks the constructor parameters it names exist. It found **34** that do
  not, all of them the agent: `orchestrator` (renamed to `model`, in 13 pages
  including the README), `max_steps` / `max_tokens` / `max_attempts` /
  `done_when` (folded into `stop_when`), `memory`, `store`, `executors`
  (dropped), `persona`, `allow_spawn` (now `team`).

  Every one of those decisions is written down in
  `docs/design/default-agent.md`, in full, with reasons. The design record was
  kept and the pages that teach the API were never touched — so the agent
  documentation describes a library that has not existed since `2.0.0`, and a
  reader copying the README's agent example gets a `TypeError` on line one.

  Held by a ratchet at 34, the way the parameter matrix holds silent cells:
  lowered as pages are fixed, never raised, with a test that fails when it is
  left slack. The sweep itself is the next release, not this one — a security
  fix should not wait behind a documentation pass.


## [2.5.2] — 2026-09-10

**Tool calls stream.** 2.5.0 shipped streaming with this named as missing and
2.5.1 made the gap loud — a tool-calling turn was not streamed and said so.
This closes it. A patch rather than a minor because no public API changed and
the release that opened the gap is the one that documented it.

### Added

- **A call is reassembled from its fragments**, on all three families. It does
  not arrive as a call: an id and a name land once, the arguments trickle in
  as pieces of a JSON string, and two calls in one turn interleave. Three
  things make the difference between working and appearing to work, and each
  is a test:

  * **Keyed on the provider's index, never on arrival order.** OpenAI numbers
    its calls and Anthropic uses the content-block index precisely because
    order identifies nothing once there is more than one. Keying on order
    splices two calls' argument strings into one that does not parse — and
    unparseable arguments recover to an empty dict, so the tool runs, with
    nothing, and the run carries on.
  * **Concatenated, then parsed once.** Parsing on the way sees truncated
    JSON on every fragment but the last. Parsing is left to the same builder
    the buffered path uses, so a malformed argument string fails identically
    whether it was streamed or not.
  * **A later fragment does not erase the name.** The name comes once and the
    fragments after it carry arguments alone; writing each fragment's empty
    name over the stored one leaves a call nothing can route.

  Anthropic streams the arguments under `input_json_delta.partial_json`, which
  is why the text filter never saw them. Google fragments nothing — its
  `functionCall` arrives whole with `args` already an object — and is handled
  by the same assembler rather than a special case.

- **Prose beside a call** is yielded as it arrives and kept on the request's
  `text`, so it is neither shown twice nor lost. The call itself is never
  yielded: the pieces are what a caller prints, and a decision is not prose.

### Notes

- **Agent turns are still not streamed**, and not for want of the capability.
  `Agent` calls the model with two retries, on the finding that a loop which
  does not retry pays for each transient blip with a lost decision — and a
  stream cannot retry once the caller has seen the first piece. The event
  stream is what `Agent.stream()` offers; text is `Skill.stream()`'s.


## [2.5.1] — 2026-09-10

Three defects in 2.5.0's streaming, found by asking the code what it does
rather than by reading what the release notes said it did. All three returned
something plausible, which is why none of them failed a test.

### Fixed

- **Every Qwen stream crashed.** `QwenClient` and `RecraftClient` override
  `build_request` *without* a `tools` parameter, and the streaming builder
  forwarded one — `TypeError`, not the `NotImplementedError` the fallback
  catches, so it reached the caller. A keyword a sibling does not take is not
  a keyword this family can pass.

- **Recraft claimed it could stream.** It renders images and has nothing to
  deliver progressively; it inherits from the client that streams, and the
  capability came with it. That is the defect the option layer was cleared of
  in 2.3.0 — a claim not backed by effect — arriving through the class
  hierarchy instead of through the data. A test now builds a streaming
  request for every provider that declares the capability, so the flag has to
  be earned rather than inherited.

- **A streamed tool call vanished.** With tools declared, a reply's call
  arrives as deltas of a JSON argument string spread over several events, and
  nothing reassembled them: the stream yielded nothing and `last_result` was
  an empty string, which reads as "the model said nothing" — the most
  expensive wrong reading available. Reassembly is a feature of its own and
  is not in this patch; until it exists, a tool-calling turn is not streamed,
  the answer comes back whole with the call intact, and a `declined` says so.
  The object is kept rather than rendered: a tool call turned into text and
  parsed back is not a tool call.

  Its repr is not streamed either — `str(result)` would have put
  `ToolCallRequest(calls=(...))` in front of whoever was printing the pieces.

- **The non-streaming fallback read Google's usage as zero**, the same
  envelope-versus-block mistake fixed in the streaming path itself, in the
  one branch that had no test.


## [2.5.0] — 2026-09-10

**Streaming.** `run()` is untouched — this is a second way to spend the same
request, not a replacement. `2.4.0` is skipped over rather than cancelled;
judging and guardrails keep their number and their content.

### Added

- **`Skill.stream()`** yields the answer as the provider produces it. The
  whole text is assembled as well as yielded — `last_result`, parsed when the
  output format asks for JSON — so a caller does not have to choose between
  showing progress and having the value. Usage is asked for explicitly and is
  `None` when the provider reported none.

- **`Agent.stream()`** yields each event as the run produces it, and
  `last_result` holds the `AgentResult` at the end. It is the **same loop**
  walked instead of exhausted, not a second implementation, and the events are
  the ones already going to hooks — so a caller's own hooks still fire, in the
  same order, and `llm_call.*` (emitted by `Skill`, not by the loop) arrives
  without anything being wired for it.

  Events, not tokens, and deliberately: a turn in an agent loop is usually a
  tool call rather than prose, so token deltas would be empty for most of a
  run and interleave with decisions in no useful order. The granularity is the
  turn, because the loop is synchronous by design — no threads, no async, the
  target is Lambda.

- **Nine providers stream** — OpenAI chat completions and everything
  compatible with it (DeepSeek, Kimi, Qwen, xAI, Perplexity, self-hosted),
  plus Anthropic and Google.

### Fixed

Four defects found by the tests rather than by the author, each returning
something plausible:

- **Google streams priced themselves at zero.** `stream_usage` returned the
  bare usage block, which two families' shapes were understood from and
  Google's was not. It returns a response-shaped envelope now, so a streamed
  report reaches the same branch of `extract_usage` a buffered one does.
- **Nothing streamed through Google at all.** The image guard tested whether
  `modalities` was present rather than what was in it, and a `Skill` sets
  `modalities: ["text"]` on every text call — so every Gemini stream declined
  with "an image is not delivered progressively", a sentence plausible enough
  to be believed.
- **Anthropic streams under-billed by the whole prompt.** Usage arrives in two
  events — input tokens in `message_start`, output tokens in `message_delta` —
  so keeping the last report drops the input side and leaves a believable
  number behind. Reports are merged, not replaced.
- **A `declined` notice was warned about and then vanished** from the
  machine-readable half, which is the half a measurement run reads: on the
  non-streaming fallback path `_built` overwrote `last_adaptations`.

### Notes

- **A provider that cannot stream still answers.** Image endpoints have
  nothing to deliver progressively and two of OpenAI's own paths speak a
  different event vocabulary; rather than raise — a library that raises on a
  provider gap is one you cannot swap a provider under — the whole answer
  arrives in one piece and a `declined` is recorded. One chunk at the end
  looks like a working stream, so `Model(on_unsupported="requirements")`
  raises for a caller who needs the real thing.
- **Streamed calls do not retry and do not fall back.** Both work by
  discarding the attempt and starting again, which is impossible once the
  caller has seen the first piece.
- **Not in scope:** `Chain` and `Pool`. A stream through a Chain means a
  stream of *steps*, which is a different question; through a Pool it means
  merging several, which is meaningless without concurrency. Half of each
  would have been worse than neither.


## [2.3.0] — 2026-09-09

**Sampling and every other option, audited.** The library's promise is one
vocabulary across providers; what it was doing was dropping whatever a
provider could not take and saying nothing. Measured rather than assumed —
by building each request twice, with and without an option, and diffing the
bodies — **69 of 116 (option, model) pairs were being changed in silence.**
For a library whose own measurement rule is "one variable between arms", that
is the mechanism that invalidates its users' comparisons.

### Added

- **The library says what it had to change.** Providers disagree, so a
  universal option cannot always travel unchanged — `top_k` has no field on
  the OpenAI wire, a reasoner refuses `temperature`, and asking DeepSeek to
  think routes to a different model. Adapting is right; adapting in silence
  is not.

  Every option now meets one of five fates — translated, adapted, declined,
  swapped, refused — and the last four are reported twice: a warning once per
  (model, option), and a record that survives on `Skill.last_adaptations`,
  `Model.last_adaptations` and `AgentResult.adaptations`, so a run can write
  down what its arms actually sent. Of the 116 pairs, none are silent now;
  the matrix has since grown to 174 and holds at zero.

  Three were not silent but wrong — `cache_control` on Anthropic and `size`
  on BFL and Qwen were reported as "this provider has no such control" while
  being honoured under another name. The conformance test now requires a
  notice whose kind matches what happened, and which names the replacement
  when there is one.

- **What each provider accepts is declared in its data**, not scattered
  through `if provider == "..."` branches: `[provider.options] accepts` and
  `format_accepts` per provider, with the two vocabularies in
  `models/_options.py`. A typo and a genuine capability gap used to read
  identically; now an unknown name is refused with the real names listed,
  and a missing control is declined with what to reach for instead —
  `top_k` on OpenAI says "top_p narrows the sampling in a way every provider
  takes".

- **A level means the same thing on every provider's own scale.** `medium`
  becomes 10000 thinking tokens on Anthropic, 8192 on Google, `"medium"` on
  OpenAI, a boolean on Qwen, and a different model on DeepSeek. The maps live
  in the provider data (`[provider.options.map.<option>]`) and are printed on
  the docs page, because a universal name without its scale is half a promise.

- **A value this model does not take fails before the wire**, not after the
  round trip is spent — an enum raises with the allowed values named, a range
  clamps and says so.

- **`Model(on_unsupported=...)`.** An option that cannot be honoured is
  reported and the request still goes; that is right for `temperature`, whose
  loss gives a different answer, and wrong for `seed`, `size`, `background`,
  `fidelity`, `output_format` and `compression`, whose loss gives a result
  that *looks* correct and silently is not. The library cannot pick — the
  cost belongs to the caller — so: `"warn"` (default, providers stay
  swappable), `"requirements"` (raise for the invisible losses only) or
  `"raise"`. The record is written before the raise.

- **`Model.effective_options`** — what this model actually runs with, asked
  for or not. For a measurement run to record beside its results.

- **`docs/reference/parameters.md`**, generated by `scripts/parameters.py`
  together with the conformance snapshot, so the page cannot say one thing
  while the code does another. Four sections: what you get without asking
  (with the author of every default), what you can ask for, what each
  provider declares it takes, and what happens per (option, model) cell.

### Fixed

- **Four providers read no image shape control at all**, and it was a gap
  rather than a limitation in every case. Google and xAI now take a ratio
  (16:9 returns 1344x768 and 1280x720, verified by decoding the pixels);
  Recraft, Qwen and OpenAI took pixels and ignored a ratio. All six accept
  either spelling now and say which they converted.

  The rest of the cells were genuinely "cannot", each established **by
  effect** rather than by status code, because three of these APIs answer
  200 to parameters they ignore and one answers 200 to invented ones:
  `seed` works on BFL and nowhere else; `top_k` exists on Anthropic and
  Google only; `output_format` is not Recraft's; Recraft's edit path derives
  its size from the input image.

- **An image key on a text model now says so** instead of blaming the
  provider: a mistake that is a mistake everywhere is refused locally, while
  one that is only wrong here is absorbed and reported.

- **`strength`, `input_fidelity` and `reference_fidelity` were one axis under
  three names** — merged into `fidelity`, direction measured rather than
  assumed. Old names keep working and report the rename.
  `output_compression` is now `compression`.

- **A multipart body no longer reports a delivered option as lost.** A form
  field carries every value as text, so `strength=0.4` went out as `"0.4"`
  and was read back as declined.

### Changed

Two defaults were the library disagreeing with a provider quietly. Both are
withdrawn; callers who want the old behaviour pass it as an option.

- **`deepseek-*` now sends `temperature=1.0`** (was `0.0`). 1.0 is DeepSeek's
  own default; 0.0 is their advice for code and maths, and adopting it here
  made one provider deterministic while no other one was — the odd arm in any
  comparison that set nothing.
- **`qwen-max` now sends `max_tokens=8192`** (was `2048`). 2048 is a quarter
  of what the model produces and was the smallest in the table by four times,
  so long answers were being cut off by us and read as the model stopping.
- **An unknown model option raises.** `Model(options={"temperatur": 0.5})` was
  accepted, ignored, and cost a run to notice. Output-format keys stay a
  warning, because that vocabulary is open — providers read names of their own.
- **Provider defaults carry their author.** Every number sent on every call is
  now marked in `[provider.default_notes]` as the vendor's or ours, and a test
  fails when one arrives anonymous. The reasons existed in the per-provider
  modules and were lost by the 1.2.3 migration, which changed no behaviour and
  dropped every justification.

## [2.2.2] — 2026-09-09

### Fixed
- **Vision works on the Responses API.** The two OpenAI wire formats name
  content differently — chat takes `text`/`image_url` with the image as an
  object, the Responses API takes `input_text`/`input_image` with the URL as a
  bare string — and the chat encoder was reused for both. Any message with
  more than one part was refused: `Invalid value: 'text'. Supported values
  are: 'input_text', 'input_image', …`. It had never worked on those models;
  the tool-returns-an-image feature of 2.2.0 only made it visible.
- **A lone text part is no longer collapsed to a bare string.** That shortcut
  is why the above went unnoticed: single-part messages are the commonest
  shape by far, so every ordinary run passed and only multi-part ones failed.

## [2.2.1] — 2026-09-09

### Fixed
- **A parameter one model refuses no longer fails the request.**
  `input_fidelity` sent to `gpt-image-2` came back HTTP 400 and the library
  added nothing to the provider's refusal. Models that reject it now carry
  `rejects` in the provider data and it is dropped before sending, with one
  warning per model and parameter — dropping a setting the caller asked for
  is not something to do quietly.

  Which models refuse it was established by asking the API, not by reading
  the guide: the guide names only `gpt-image-2`, while both GPT Image 2.5
  models and `gpt-image-1-mini` refuse it too. `gpt-image-1.5`, `gpt-image-1`
  and `chatgpt-image-latest` accept it and still receive it.

## [2.2.0] — 2026-09-09

Providers disagree, and the library absorbs the difference instead of the
caller. Everything here was reported from building on 2.1.0.

### Fixed
- **`Agent` could not start on Google.** The synthetic `pool` tool declared an
  array parameter with no element schema, which Google rejects outright —
  before the first step, and along with every user tool in the same request.
  The Google sanitizer now fills in a permissive default for any array missing
  `items`, so a tool that works elsewhere is not the reason a run cannot begin.
- **A response cut off at the token ceiling is no longer a `JSONDecodeError`.**
  It arrives as HTTP 200 carrying JSON that stops, and only the OpenAI chat
  path recognised it; Google and the Responses API did not. All three now
  raise `TruncatedResponseError`, naming the reason and the token count that
  was hit — the number `max_tokens` has to be raised above.
- **Structured replies are checked against the schema.** Google strips
  `additionalProperties`, which inverts "no fields beyond these": models
  answered with fields the schema forbade and nothing noticed until two
  providers' outputs were diffed by hand. A mismatch now raises
  `InvalidStructuredOutputError` carrying the list of violations.

### Added
- **`portable_schema(schema, target)`** — the same schema in the form
  `"openai"` (strict) or `"google"` accepts. The two families demand opposite
  things of the same document, and every caller was deriving this by hand. A
  map with arbitrary keys has no strict equivalent and is refused by name
  rather than mangled.
- **`check_structure(value, schema)`** — a dependency-free structural check:
  declared types, required properties, closed objects, enums, array elements.
  Shallow on purpose; a full validator is a dependency and this package has
  one.
- **`TruncatedResponseError` and `InvalidStructuredOutputError`**, both also
  `ValueError` so existing handlers keep working.
- **A tool can hand back an image.** Media parts are carried through instead
  of serialised, so an agent that renders something can look at what it
  rendered — a vision loop no longer has to be lifted out of the agent into a
  separate Skill. Anthropic takes the image inside the tool result; OpenAI and
  Google receive it as the user message immediately after, captioned with the
  call it came from, and the result text says what follows.

  Two layers had to stop flattening it, and the second was only found by
  running it: `result_message` json-dumped every dict and list one level
  above the code that knows how to carry media, so the first version passed
  its unit tests and shipped an agent that sent base64 as prose. Measured on
  the real path, sighted against a blind arm that receives the same words
  without the picture, three trials each: gpt-4o-mini 3/6 → 6/6,
  gemini-2.5-flash 0/6 → 5/6, blind 0/6 throughout.
- **A warning when Google strips `additionalProperties`**, once per process,
  raised where the schema is written rather than where the wrong answer
  appears.

### Added, 2026-09-09
- **GPT Image 2.5** — `gpt-image-2.5-flare` and `gpt-image-2.5-sunburst`.
  Flare is the default choice for most work (better than `gpt-image-2` at half
  the latency); Sunburst is the premium end, for edits needing tight control.
  The docs read the other way round, so the distinction sits beside the
  entries.
- **`output_compression`** reaches the wire on generations and edits. It was
  documented and silently dropped.

### Fixed, 2026-09-09
- **Image tokens sent in are priced at their own rate.** $8 per million
  against text's $5 on the 2.5 family. One input rate and one input count
  billed an edit carrying reference images as if it were prose: measured on a
  live call, 37% under on the input line and 22% under across the call.
  `image_input` is an optional price field, `image_input_tokens` a separate
  count.
- **Google's thinking tokens are counted.** `thoughtsTokenCount` sits outside
  `candidatesTokenCount` and is billed — "response pricing is the sum of
  output tokens and thinking tokens". A live vision turn reported 1 output
  token beside 235 of thought.

### Changed
- **Media sources resolve on send, not at construction.** `{"kind": "file"}`
  used to be read into base64 inside `Skill.__init__`, and `substitute()` only
  ever reached text parts — together they pinned a vision Skill to one file
  for its lifetime. A media path may now carry a `{placeholder}`, so one Skill
  serves a whole `Pool` of images.
- **`Pool` records usage per item** and sums it on `pool.usage`. Reading
  `last_usage` off the shared runner raced: whichever thread finished next
  overwrote it.
- **A `Chain` with a `Wait` step and no store warns at construction.** The
  pause is answered in another process, usually a separate command and
  sometimes the next day; the in-memory default said so only at `resume()`
  time, as "No suspended run in the store".
- **`Tool` documents what a returned dict does.** The keys are spread into the
  run's variables and the step name is not used — deliberate, since one step
  can feed several named inputs, but undocumented until now and discovered
  only through a wrong result.

## [2.1.0] — 2026-08-06

Eval — the fifth and last of the planned mechanisms. Additive; nothing
existing changes.

### Added
- **`yait_aichain.eval`** — run the same cases through several arms, several
  times each. `Eval` executes arms × cases × trials to a durable JSON-lines
  ledger and resumes from it; `Report` counts.
- **Three columns, because they disagree.** `mean` (share of attempts passed),
  `pass^k` (share of cases passed on *every* trial) and `flips` (share of
  cases answered differently on different trials). On a regression run the
  reference agent led on mean and trailed on `Pass^3`; `flips` is the shape
  neither of the other two shows.
- **`Report.of(rows, arm=…, case=…, ok=…)`** reports data produced elsewhere,
  so a benchmark that owns its own loop still gets `Pass^k`, stratified
  tables, paired tests and validity guards without being rewritten.
- **Scorers** — `exact`, `contains` (with partial credit), `regex`, `numeric`,
  `all_of`, and `judge` for an LLM verdict. A judge reply that will not parse
  raises rather than defaulting to a pass.
- **Controls** — `Report.controls(oracle, noise)`: feed the harness
  known-correct and known-wrong answers; if the oracle does not score ≈1.0 the
  metric is broken rather than the system.
- **Validity guards** — `Report.reject()` drops a cell with too many empty
  outputs or errors. A rejected cell is *absent* from every figure, not zero.
- **`Report.paired(a, b)`** — an exact sign test on discordant cases, and it
  says when there are too few to conclude anything. `se(p, n)` beside it.

### Fixed
- **harmony replies no longer lose their tool call.** gpt-oss models answer in
  OpenAI's channel format; a server that does not implement it — mlx_lm,
  llama.cpp, vLLM without a tool-call parser — passes the raw text through, so
  the call arrived as a string and a string is a final answer by our own
  contract. The `analysis` channel is dropped rather than surfaced as the
  answer, `commentary to=functions.NAME` becomes a typed call, `final` is the
  text. A body cut by `max_tokens` yields empty arguments instead of an
  exception.
- **`scripts/release.sh` works again.** It read `project.version` from
  `pyproject.toml`, which has not existed since the version became dynamic in
  1.4.1 — the script failed with a `KeyError` on its first step.

## [2.0.0] — 2026-08-03

The agent is rebuilt as a conversation that can act, on native tool
calling. Breaking across the agent surface; the model layer gains a tool
channel without changing existing call sites.

### Changed
- **`Agent` is one loop.** `Agent(model, tools, instructions, mode, team,
  stop_when)`; a text reply is the answer, requested calls execute and
  append, the conversation only grows (stable cacheable prefix).
  `orchestrator=` is now the first positional `model`.
- **Native tool calling.** Schemas ride the provider `tools` field; replies
  parse to typed `ToolCall`s. Four wire shapes golden-tested: chat
  completions, Responses API, Anthropic blocks, Google functionCall.
- **`stop_when` replaces `max_steps`/`max_attempts`/`max_tokens`/`done_when`.**
  `check()` that passes is success; a ceiling reached is failure;
  `AgentResult.stopped_by` names which fired.
- **Reasoning levels validate against the provider's `reasoning_map`**;
  openai gains `"none"`. Providers without a map refuse the option loudly.
- **`Pool` refuses a fan-out it cannot feed** — pre-flight, in the caller's
  thread, instead of returning a silent list of `None`.

### Added
- `planner_model` (waterfall): think while planning, act without
  deliberating — measured 0.85 vs 0.70/0.65 for always/on-demand reasoning
  on τ²-bench retail with everything else leveled.
- Synthetic tools `pool`, `delegate`, `write_plan` on the native channel;
  `team=None | [workers] | "auto"` gates delegation.
- Externally driven mode: `opening()` / `step()` / `execute()` — one
  decision per call, for harnesses and serverless drivers.
- `AgentResult.cost` and `stopped_by`.

### Removed
- `mode="goal"` (the loop without a plan is now `mode="agile"`, the
  default), `done_when`, `max_attempts`, `allow_spawn`, `executors`,
  agent suspend/resume, and the agent-memory preview layer — state lives
  in the conversation.

---

## Before 2.0, incompletely

This file records twelve releases; the repository carries twenty-eight tags.
Everything from `1.2.5` to `1.6.0` — seventeen releases — shipped without an
entry, and the 1.6.0 text below was recovered from the README, which was
carrying it alone.

The gap is written down rather than filled: reconstructing seventeen summaries
from commit history would produce a record of what the log says, not of what
was released, and a changelog nobody can trust is worse than one that names
its own holes. What is enforced instead is that it stops here — a release
whose version has no entry fails `tests/test_changelog.py`.

## [1.6.0] — 2026-07-27

**Goal mode** — a third agent mode, for tasks where no plan can be written in
advance because each step depends on what the last one returned.

```python
agent = Agent(orchestrator=Model("claude-sonnet-4-6"), tools=[...],
              mode="goal", done_when=lambda memory: "answer" in memory)
```

- No planning phase. Each iteration the orchestrator sees the objective, the
  observation trail, and what has been ruled out — and picks one action.
- `done_when` is **required** and is best given as a callable: a callable lets
  the run finish on a `check` the harness performed, a string only ever on the
  model's own claim. An unearned `final_answer` is recorded as `refuted` and the
  loop continues.
- Stop rules an open-ended loop needs: `done_when` met, iteration cap (50),
  token budget (250 000), and **no progress** — five consecutive failed attempts
  end the run instead of spending the rest of the budget proving it again.
- Suspend/resume, permissions, checkpoints and the journal all work unchanged.

**Journal entries now carry the observation.** A record of what was *attempted*,
without what it *returned*, cannot drive a next decision — an agent that cannot
read its own feedback re-issues the same action forever. Entries keep a bounded
excerpt of the result (the full value stays in memory), and the trail rendered
into the prompt includes failures, since a failed probe still returned
information. Applies to all modes.

**`memory_read` — a built-in tool on every agent.** Memory previews in the
prompt are truncated so one large value cannot crowd out everything else, but
until now there was no way past that limit: an agent could store a document,
see its first 500 characters on every subsequent turn, and never reach the
rest. Measured on a research task, that failure is total — the agent searched
15 times, stored 48 sources and never wrote an answer, because it could not
read what it had gathered. `memory_read(key, offset, length)` pages through a
stored value, and a truncated preview now says so and names the tool. Prompt
stays bounded; access does not.

**Three fixes found by running the agent, not by reading it.**

- Spawning from a goal-mode agent raised `ValueError`. `spawn()` forwarded
  `mode` but not `done_when`, and goal mode requires one — so `allow_spawn=True`
  crashed the moment the orchestrator delegated. A child now falls back to the
  plan-driven mode: `done_when` is a predicate over *this* agent's objective and
  memory, and neither transfers to a scoped sub-task.
- A spawned child could read its **parent's** memory. `spawn()` forwards the
  parent's tool list, which now contains an agent-bound `memory_read`; the child
  prepended its own and ended up with two tools of the same name, one pointed at
  the wrong memory. A foreign reader is dropped on construction.
- Reading memory wrote it back. The result of a `memory_read` was eligible for
  `store_as` like any other, so looking at a stored document copied it under a
  second key — measured on a research run as 9 of 13 reads going to the agent's
  own bookkeeping rather than to sources. A memory view is no longer stored.

**Repetition detector** — `has_progress()` catches a run that is failing; it
cannot catch one that is succeeding pointlessly. `Journal.is_repeating(k)`
reports a window of attempts that were all the same move, and goal mode writes
that into the next action prompt. Surfaced, never enforced: an agent circling a
hard sub-problem must not be cut off, so the model decides. Only exact
repetition counts — a similarity-based rule was measured against real runs and
did not separate a stuck agent from a healthy search.

**Fix — `Journal.has_progress()` judged on a partial window,** reporting "no
progress" after a single early failure. It now requires a full window, so one
bad attempt cannot end a run that was about to recover.

See [`examples/22_goal_mode.py`](examples/22_goal_mode.py) and
[docs/agents/overview.md](docs/agents/overview.md#goal-mode).

### 1.5.2

**The attempt journal — an append-only record of what the agent did.** Separate
from `AgentMemory` (which holds the data the agent works *with*), the journal
records every attempt with a typed outcome and the evidence behind it, and is
preserved across suspend/resume.

```python
res = agent.run("…")
for e in res.journal:
    print(e["outcome"], e["evidence"]["kind"], e["intent"])
# failed  check        call the API      ← we KNOW it failed: the tool raised
# done    model_claim  fallback path     ← the model says so; nothing verified it
```

- **Outcomes:** `done` · `failed` · `refuted` · `skipped`.
- **Evidence is typed** — `check` (a programmatic fact) vs `model_claim` (asserted,
  not verified). A run whose `done` entries are all `model_claim` has proven
  nothing, and the journal shows that instead of hiding it.
- **Do not redo** — `refuted` entries are fed back into the next action prompt as
  an `ALREADY RULED OUT` block, so a long run stops re-attempting dead ends.
- **Stuck detection** — `Journal.has_progress(k)` answers "did anything actually
  move lately", the stop rule an open-ended loop needs (a budget alone lets an
  agent spin until the tokens run out).

**Crash recovery.** The run is now checkpointed to the `Store` after **every
committed step**, not only when it suspends — so an unplanned death (crash, OOM,
function timeout) is recoverable: a fresh `Agent` sharing the store picks the run
up with `resume(run_id)`, restoring memory, cursor and journal. Suspend/resume
becomes the special case of the same mechanism. (Recovery retries the step that
was in flight — gate side-effecting tools with a `PermissionPolicy` or make them
idempotent; committed steps are never re-run.)

**Fix — a run that suspended twice became unresumable.** Since 1.4.4 the parked
`run_id` is kept aligned with the event-stream id, but `resume()` still deleted
that id unconditionally afterwards — so a second suspend parked a document and
immediately dropped it. Any approval → resume → approval flow lost the run.
`resume()` now keeps the document when the loop re-suspended under the same id.

**Fix — honest success could be bypassed.** The 1.3.4 guarantee ("a step that
ended with an execution error fails the run") was only enforced on the
run-to-completion path: an orchestrator emitting `final_answer` skipped the check
and the run reported `success=True` despite an earlier failed step. The same rule
now applies on every exit path.

### 1.5.1

**Recraft raster → vector (vectorize).** `Model("recraft-vectorize")` traces an
existing raster image into an SVG — no prompt, no generation, just a format
conversion (distinct from `imageToImage`, which is a content *variation*). Same
`Model` + `Skill` ergonomics; the model name selects the `/v1/images/vectorize`
endpoint. Verified live (returns `{"mime_type": "image/svg+xml", ...}`).

```python
Skill(model=Model("recraft-vectorize"),
      input={"messages": [{"role": "user", "parts": [
          {"type": "image", "source": {"kind": "file", "path": "logo.png"}}]}]},
      output={"modalities": ["image"], "format": {"type": "image"}}).run()
```

**11 providers / 82 models.**

- Anthropic json-mode robustness: when the model wraps its JSON in prose (leading
  or trailing commentary), the first balanced top-level object/array is now
  recovered instead of failing to parse (`_extract_first_json`).

### 1.5.0

**Directed multi-turn reasoning in one `Skill`.** Some tasks are a guided
sequence of refinements — ask, refine, refine, finish — where *you* know the
steps (unlike an `Agent`, which decides them). Express it inside a single Skill:
put several `user` turns in the message template, separated by **"generate here"
markers** — an `assistant` turn with no `parts`. Each marker is one model call;
its reply is appended to the running context, so later turns see what earlier
ones produced. No `Chain`, no manual output threading.

```python
Skill(model=Model("claude-sonnet-4-6"), input={"messages": [
    {"role": "system",    "parts": ["Be concise."]},
    {"role": "user",      "parts": ["Write 10 quotes about {topic}."]},
    {"role": "assistant"},                                   # ← generate, keep in context
    {"role": "user",      "parts": ["Drop the 5 most clichéd, write 5 fresh."]},
    {"role": "assistant"},                                   # ← generate, keep in context
    {"role": "user",      "parts": ["Translate the result into {language}."]},
]}).run(variables={"topic": "perseverance", "language": "French"})
```

- An `assistant` turn **with** `parts` is a fixed/seed turn (few-shot, a default
  answer) — no model call. **Without** `parts` it's the generate marker.
- The final turn uses the skill's `output` format (text / json); intermediate
  turns are plain text (they only feed the context).
- `skill.history` holds each generated turn (`history[-1]` is the return);
  `skill.last_usage` is the sum across turns.
- **Backward compatible:** no markers → one call, exactly as before.
- Structure is validated at construction (`ValueError`): at least one `user`
  turn, no leading `assistant`, no two `assistant` turns in a row.

### 1.4.6

**Honest billing errors + registry hygiene.**

- New **`InsufficientCreditsError`** (subclass of `APIError`). An out-of-credits
  state is a *billing* problem, not a credentials problem, but providers signal
  it inconsistently — xAI as `403 permission-denied "used all available credits"`,
  Reve as `402 PARTNER_API_BUDGET_EXHAUSTED`, OpenAI as `429 insufficient_quota`,
  Anthropic as `400 credit balance is too low`. These used to surface as
  `AuthenticationError` / `RateLimitError` / `InvalidRequestError`, sending you to
  re-check the key. Now they are detected from the response body (checked before
  the status-code mapping) and raised as `InsufficientCreditsError`, which is
  **never retried** (retrying the same account won't add funds).
- Registry hygiene (model names verified live against each API): removed the
  phantom `grok-imagine-image-pro` (xAI rejects it) and added the live-verified
  Recraft **V4.1 variants** — `recraftv4_1_utility` (controlled raster for
  product/icon/e-commerce), `recraftv4_1_utility_pro`, `recraftv4_1_pro`,
  `recraftv4_1_utility_vector`. **11 providers / 81 models.**
- `registry.refresh("perplexity")` now works: Perplexity exposes a public
  `/v1/models` router catalog, so `list_models()` reads it live (falling back to
  the curated static list only if the call fails) instead of always returning a
  hard-coded list.

### 1.4.5

**+1 image provider: Reve** (`api.reve.com`). `Model("reve-image")` generates,
edits, and remixes images behind the same `Skill`. Reve has its own (non-OpenAI)
wire shape, so it is its own client family: the operation is chosen by the input —
no image → `POST /v1/image/create`; one input image → `/v1/image/edit`
(`edit_instruction` + `reference_image`); two or more → `/v1/image/remix`
(`reference_images[]`, with `<img>0</img>` tokens in the prompt). Bearer auth
(`REVE_API_KEY`), `aspect_ratio` and `version` (default `latest`) forwarded.
**Now 11 providers / 78 models.**

```python
Skill(model=Model("reve-image"),
      input={"messages": [{"role": "user", "parts": ["A stack of golden pancakes"]}]},
      output={"modalities": ["image"], "format": {"type": "image", "aspect_ratio": "16:9"}}).run()
```

### 1.4.4

**The step boundary — observability & control.** Make every step of an Agent,
Chain, or Skill legible and governable from *outside* the model, without
touching `run()`.

```python
from yait_aichain import Tracer, PermissionPolicy
from yait_aichain.tools import Tool, FINANCIAL

class IssueRefund(Tool):
    name = "issue_refund"; risk = FINANCIAL          # ← risk class as data
    ...

tracer = Tracer()
agent = Agent(
    orchestrator = Model("gpt-4o-mini"),
    tools        = [IssueRefund()],
    hooks        = [tracer],                          # structured event stream
    permissions  = PermissionPolicy({"financial": "approve"}),
)
result = agent.run("Refund order #123")              # pauses for approval
result = agent.resume(result.run_id, signal={"approved": True})
for e in tracer.events: print(e)                     # run/step/llm_call/tool_call
```

- **Logging instead of `print()`.** The library emits through named
  `logging` loggers (`yait_aichain.*`) with a `NullHandler`; the application
  decides the sink. `verbose=` still prints to the console (now routed through
  the logger). Nothing is printed by default.
- **Lifecycle hooks + events.** `Agent`, `Chain`, and `Skill` accept
  `hooks=[...]` — callables that receive a structured `Event` at every boundary
  (`run.*`, `step.*`, `llm_call.*`, `tool_call.*`) carrying `run_id`, `step`,
  token `usage`, `cost`, `duration`, `error`. Ships `Hook`, `Tracer`, and
  `LoggingTracer` conveniences. A buggy hook can never crash the run.
- **Permission matrix.** A tool declares a `risk` class (`read`/`draft`/`write`/
  `external`/`financial`/`destructive`/`privileged`); a `PermissionPolicy` maps
  it to `allow` / `approve` / `deny` enforced *before* the tool runs, outside the
  model. `approve` pauses for an external approval (reusing suspend/resume —
  no manual `Gate`); `deny` still returns a result. Opt-in: no policy → no gating.
- **Tool-call repair.** Tool arguments are validated locally against the schema
  before execution; a malformed call returns a model-readable remediation so the
  agent corrects it within the step's attempt budget.
- **Invariant:** every tool call returns a result — even on denial, validation
  failure, or error.

### 1.4.1

Version is now a **single source of truth** in `yait_aichain/__init__.py`
(`__version__`); `pyproject.toml` reads it dynamically (`[tool.setuptools.dynamic]`)
and the publish workflow checks the same file — no more drift between the two
(which silently blocked the 1.4.0 publish until both were bumped).

### 1.4.0

**Multimodal: image-to-image (image editing).** Restyle / recompose / edit an
existing image across four providers behind the same `Skill` — an input image
part plus an image-output model *is* an edit; swap the provider by changing the
model name, nothing else.

```python
Skill(
    model  = Model("gpt-image-1.5"),     # or gemini-3.1-flash-image / grok-imagine-image / qwen-image-edit
    input  = {"messages": [{"role": "user", "parts": [
        {"type": "image", "source": {"kind": "file", "path": "product.png"}},
        "Place this product on a marble kitchen counter, soft morning light",
    ]}]},
    output = {"modalities": ["image"], "format": {"type": "image"}},
).run()
```

- **Six image providers** edit: OpenAI (`gpt-image-*`, `chatgpt-image-latest`,
  multipart `/v1/images/edits`), Google (Gemini image, conversational edit), xAI
  (`grok-imagine-*`, JSON edits), Qwen (`qwen-image-edit` series, synchronous
  multimodal-generation), plus two dedicated image houses — **Recraft**
  (`recraftv3`, multipart imageToImage) and **Black Forest Labs / FLUX**
  (`flux-kontext-*`, async submit→poll→download). New `image-to-image` task in
  the registry.
- **+2 providers, now 10 / 77 models.** Recraft (`RECRAFT_API_TOKEN`) and BFL
  (`BFL_API_KEY`) join as image specialists — `Model("flux-kontext-pro")` /
  `Model("recraftv3")` resolve and edit like any other.
- **Multiple reference images for edits.** Pass several image parts to compose /
  restyle with references — OpenAI (`image[]`, ≤16), xAI (`images`, ≤3), Qwen and
  Gemini all accept multi-image input.
- **Empty image responses raise a clear error.** When a provider returns no image
  (blocked / refused / text-instead-of-image), the call raises a descriptive
  `ValueError` instead of silently returning `base64=None` (which crashed callers
  on decode).
- **Local files as input.** A media source `{"kind": "file", "path": "..."}` is
  read and base64-encoded automatically (MIME inferred), so you can pass a path
  straight into any vision or edit call.

### 1.3.6

Added **Kimi K2.7 Code** (`kimi-k2.7-code`) — Moonshot's coding-focused model
($0.95 / $4.00 per 1M input/output). It runs with Thinking enabled by default
on the Kimi API; the client never disables it, so it works out of the box. For
non-coding tasks, `kimi-k2.6` stays the recommendation.

### 1.3.5

Model registry refresh — Google image models.

- Gemini image generation is now GA: `gemini-3.1-flash-image` and
  `gemini-3-pro-image` (the `-preview` suffixes are gone), plus the new
  `gemini-2.5-flash-image`. These are the migration target for Google's
  discontinued Imagen 4 endpoints (`imagen-4.0-*`), which this library never
  referenced.

### 1.3.4

Agent engine, token accounting, and transport fixes from the audit.

- **Token budget** is enforced *within* a step (after the action and execution
  calls, not only between steps), so one step can no longer overshoot
  `max_tokens`; agile replans are capped; tokens from an unparseable
  orchestrator reply are still counted.
- **Honest success**: a run fails if *any* executed step ended in an execution
  error (not only the last), and the final output is the last executed step's
  output (even `None`) rather than a stale earlier value.
- **Agent LLM calls go through the `send()` seam** (consistent with Skill;
  async providers work).
- **Usage**: `Skill.last_usage` resets each run (no stale value after a
  failure); `chain.last_usage` includes Agent-step tokens; `NetworkError` is
  retried within a model when `max_retries > 0`.
- **Robustness**: token extraction tolerates `usage: null` / non-numeric; the
  DeepSeek-reasoner gate matches the name exactly; Google embeddings accept
  `GOOGLE_API_KEY` or `GOOGLE_AI_API_KEY`.
- **Unified HTTP transport**: one `make_http()` factory builds the urllib3
  manager for both model and tool clients and honours `HTTPS_PROXY` /
  `HTTP_PROXY` — a proxy now applies to tool traffic (search, fetch, REST, …),
  not just LLM calls.

### 1.3.3

Durable-run (suspend/resume) correctness fixes from the audit.

- Resume no longer re-runs already-attempted steps: skipped/failed steps get a
  terminal status and the resume cursor skips past them; a terminal error during
  a resumed run clears the parked document, so a duplicate trigger is a no-op
  (no double side effects).
- A nested `Agent` that pauses (`Wait`/`Gate`) inside a `Chain` now propagates
  the suspension up — `chain.run()` returns a `SuspendedResult` instead of
  reporting a failed step, and `chain.resume()` continues the child agent run.
- `RunContext` is now real: exposed as `chain.context` / `agent.context` during
  a run, persisted in the run document, and restored on `resume`
  (`Agent.run` / `Agent.resume` accept `context=` too).
- `FileStore`: a non-JSON-serialisable variable/output at a suspend point raises
  a clear error instead of a raw `TypeError`.

### 1.3.2

Correctness & safety fixes from a code audit.

- Qwen image: a terminal task failure (FAILED / timeout / no result) now raises
  a non-retryable `TaskFailedError`, so a retry or model fallback can't silently
  submit a second billable generation.
- SSRF hardening: `RestApiTool` now blocks private / loopback / metadata
  targets (opt out with `AICHAIN_ALLOW_PRIVATE_URLS`); the Qwen TTS audio
  download is guarded and no longer follows redirects.
- VectorDB: `ChromaBackend.delete` refuses an empty `ids`+`filter` (which would
  wipe the whole collection), matching Qdrant.
- `FileStore`: `fsync` before the atomic rename (durable parked runs); a corrupt
  run file raises a clear error instead of a raw `JSONDecodeError`.
- `convertToHTML` writes through the confined output path, consistent with the
  other convert tools.

### 1.3.1

Image-generation fixes and additions.

- Qwen text-to-image now works. DashScope serves the `wan` models only through
  its native *asynchronous* task API, so a new request lifecycle (submit → poll
  → download) runs behind a `send()` seam on the client and returns the image in
  the same shape the synchronous path does. Registry updated to the available
  `wan2.2-t2i-flash` / `wan2.2-t2i-plus` (the old `wanx2.1-*` ids 404'd).
- New OpenAI image models: `chatgpt-image-latest` (the always-current model used
  by ChatGPT — fast, recommended default) and `gpt-image-2`.
- Transparent backgrounds for `gpt-image-*` / `chatgpt-image-*`: `background`
  and `output_format` are forwarded from the output format (`background:
  "transparent"`, `output_format: "png"`), so these models can emit real PNG
  alpha. The `response_format` param is correctly omitted for the
  `chatgpt-image-*` family (it returns base64 natively and rejects the param).

### 1.3.0

Durable, resumable runs — the serverless core. A Chain or Agent can **pause**
until an external signal arrives (a human, a webhook, a cron tick) and
**resume** later, even in a different process. All additive; existing programs
are unchanged.

- `Wait` / `Gate` suspend tools — pause a run until a signal arrives; on resume
  the signal drives the step. `Wait` is a leaf (its output is the signal);
  `Gate` wraps any tool behind an approval decision.
- `Chain.resume(run_id, signal)` / `Agent.resume(run_id, signal)` — continue a
  suspended run from where it paused; completed steps are not re-run.
- Self-contained run documents in a pluggable `StateStore`: `InMemoryStore`
  (default, process-local) and `FileStore` (survives restart); subclass for
  S3/DynamoDB/any KV. The store holds only suspended runs and a resume is
  idempotent — a duplicate trigger is a no-op.
- `run()` returns a falsy `SuspendedResult` (carrying `run_id` and `awaiting`)
  when paused instead of a final result. `RunContext` (tenant, metadata) can be
  passed to `run()` for per-request context.

### 1.2.6

- Security hardening: SSRF guard on outbound URL tools (private/loopback ranges
  blocked; opt out with `AICHAIN_ALLOW_PRIVATE_URLS=1`), URL-scheme allow-list,
  and output-path confinement (`AICHAIN_OUTPUT_ROOT`). Safe-class checks on
  tool instantiation during chain load.

### 1.2.5

- Correctness fixes across the model and tooling layer: response parsing,
  search/MCP timeouts, table chunking when a header exceeds the chunk size,
  and assorted edge cases surfaced by analysis.
- Refreshed the model registry to current provider catalogues and June 2026
  pricing.

### 1.2.3

Two-tier model layer: **format is code (by API family), provider is data.**
Changing provider — one word in `Model("…")` — changes only data, never the
request format or any model code. Behaviour is byte-for-byte unchanged
(guarded by characterisation tests).

- `Model` is a single, thin, data-driven class: it resolves the provider from
  the name and delegates the wire format to the matching family client. The
  provider is exposed as `model._provider`.
- Provider settings, model capabilities and prices live in data — one file per
  provider under `models/providers/*.toml`.
- The protocol layer (`clients/`) owns the wire format. Five family clients
  cover all eight providers: `OpenAIClient` (openai, xai, kimi, deepseek),
  `PerplexityClient`, `QwenClient`, `AnthropicClient`, `GoogleClient`.
- Removed the per-provider `Model` subclasses and client classes; use
  `Model("name")` (or an explicit `provider/` prefix for custom names).
- Fixed Qwen region endpoints (`us`, `hk`) and base URL.

### 1.2.1

- Removed leaked application code from the library (`skills/summarise.py`,
  `tools/section_context.py`) — an app-specific pipeline that did not belong in
  the general-purpose library.

### 1.2.0

Mechanism 1 — "LLM layer as data". All additive; the minimal program is
unchanged.

- `Usage` on every result — normalised `input_tokens` / `output_tokens` /
  `total_tokens` across providers; additive; `skill.last_usage` and
  `chain.last_usage` sum across steps.
- Cost estimation from a per-model price table (`usage.cost`; `None` when a
  model is unpriced).
- Exception hierarchy under `APIError` (`RateLimitError`,
  `AuthenticationError`, `InvalidRequestError`, `NotFoundError`,
  `ServerError`, `NetworkError`).
- `provider/model` routing — `Model("openai/gpt-4o")`; unlocks custom /
  fine-tuned names.
- Model fallback chain — `Skill(model=[primary, backup, …])` advances on a
  transient failure; a non-transient failure propagates immediately.
- `registry.refresh(provider)` — diffs the registry against the provider's
  live `list_models()`.

### 1.1.0

Foundation repaired: five features that never worked in the installed package
are revived, plus fragility closed across the library. 69 regression tests.

- Fixed `Chain.load()`, Agent inside `Chain`, the Qwen embedder/reranker, and
  Agent persistent memory — each previously crashed or was dead code.
- Honest `success=False` on token-budget exhaustion and exhausted step retries.
- Hardened LLM-response and agent-JSON parsing; timeouts on all search tools
  and the MCP bridge; `delete(ids=[])` no longer wipes a collection.
- Chunker contract fixes; thread-safe `Chain.run()`; fixed
  `from yait_aichain.tools import *`.
- POST retry policy (429/503); Anthropic auto-raises `max_tokens` above the
  thinking budget; gpt-5 / o-series parameter correctness; Chroma v2; Qdrant
  string IDs; batched VectorDB `upsert`; safe prompt templating.
- Security: the Google API key moved from the query string to the
  `x-goog-api-key` header.

### 1.0.0

Initial release — `Skill`, `Chain`, `Pool`, `Agent`, `Tool` / `MCPTools`,
`VectorDB`, `Reranker`, and 8 providers (Anthropic, OpenAI, Google, xAI,
Perplexity, Kimi, DeepSeek, Qwen).

## [1.2.3] — 2026-06-14

Two-tier model layer: **format is code (by API family), provider is data.**
Changing provider — one word in `Model("…")` — changes only data, never our
request format or any model code. Behaviour is byte-for-byte unchanged
(guarded by golden-master + family-equivalence characterisation tests).

### Changed
- **`Model` is now a single, thin, data-driven class.** It resolves the
  provider from the name, merges that provider's defaults from data, and
  delegates `to_request`/`from_response` to the family client that owns the
  wire format. The provider is exposed as `model._provider`.
- **Provider settings, model capabilities and prices live in data** —
  one file per provider under `models/providers/*.toml`.
- **The protocol layer (`clients/`) owns the wire format.** Five family
  clients cover all eight providers: `OpenAIClient` (openai, xai, kimi,
  deepseek), `PerplexityClient`, `QwenClient`, `AnthropicClient`,
  `GoogleClient`. Each holds both format (`build_request`/`parse_response`)
  and transport.
- Registry query (`registry.models/providers/tasks/is_supported/refresh`)
  and cost (`estimate_cost`/`attach_cost`) now read the provider data.

### Removed
- The 8 per-provider `Model` subclasses (`OpenAIModel`, `AnthropicModel`, …).
  Use `Model("name")`; for an unrecognised name use an explicit `provider/`
  prefix. The provider is available as `model._provider`.
- The 8 per-provider client classes — dissolved into the five family clients.
  `clients` now exports the family classes; the per-provider `*Client` names
  (`GoogleAIClient`, `XAIClient`, `KimiClient`, `DeepSeekClient`) are gone.
- `models/_registry.py` and `models/_pricing.py` — their data moved into the
  provider files; query folded into `models/_base`, cost into `models/_usage`.

### Fixed
- Qwen region endpoints (`us`, `hk`) and base URL — restored to the correct
  per-region hosts.

## [1.2.1] — 2026-06-14

Remove leaked application code from the library.

### Removed
- `skills/summarise.py` (`make_summarise_skill`) and `tools/section_context.py`
  (`SectionContextTool`) — two halves of one app-specific pipeline ("sectional
  reports / rolling-context"). Neither was used inside the package or covered
  by tests; `summarise.py` had broken imports (`from models import …`) and a
  docstring reference to a non-existent `run.py`. App-specific code does not
  belong in the general-purpose library — moved out. `SectionContextTool`
  dropped from `tools` exports.

## [1.2.0] — 2026-06-14

Mechanism 1 — "LLM layer as data": token accounting, cost, explicit routing,
a typed error hierarchy, model fallback, and live model-list refresh. All
additive — the minimal program is unchanged (lightness invariant).

### Added
- **`Usage`** on every result — normalised `input_tokens` / `output_tokens`
  / `total_tokens` across all providers (OpenAI / Anthropic / Google shapes
  flattened).  `skill.last_usage`; `chain.last_usage` sums across steps.
  `Usage` is additive (`a + b`).
- **Cost estimation** from a per-model price table (`models/_pricing.py`,
  USD per Mtok).  `usage.cost` is filled in; an unpriced model yields
  `None` (honest unknown).  Reference snapshot, trivial to update.
- **Exception hierarchy** under `APIError`: `RateLimitError` (carries
  `retry_after` from the header), `AuthenticationError`, `InvalidRequestError`,
  `NotFoundError`, `ServerError`, `NetworkError`.  `except APIError` still
  catches everything.  Exported at the top level.
- **`provider/model` routing** — `Model("openai/gpt-4o")` selects the
  provider by prefix and unlocks custom / fine-tuned names the auto-detector
  can't recognise (`Model("openai/ft:gpt-4o:org:abc")`).  Bare names work
  exactly as before; the prefix never reaches the API.
- **Model fallback chain** — `Skill(model=[primary, backup, …])` advances to
  the next model on a transient failure (rate limit / server / network); a
  non-transient failure (auth / bad request) propagates immediately.
- **`registry.refresh(provider)`** — diffs the static registry against the
  provider's live `list_models()` → `{new, removed, live, registered}`;
  surfaces roster drift without mutating the registry.

## [1.1.0] — 2026-06-14

The "foundation repaired" release: the audit surfaced five features that
**never worked** in the installed package — they are now revived, plus
fragility was closed across the whole library. 69 regression tests.

### Fixed
- `Chain.load()` — broken absolute imports raised `ModuleNotFoundError` on
  every call; the method now works.
- Agent inside `Chain` — detection via MRO (previously the agent step failed
  with `TypeError`; the entire agent branch of Chain was dead code).
- Qwen embedder and reranker — a broken `resolve_qwen_base_url` import crashed
  the very first `embed()`/`rerank()`.
- Agent persistent memory — `Agent.run()` wiped the state loaded from file
  before the first step; memory now persists across runs.
- The agent returns an honest `success=False` on token-budget exhaustion and
  when all retries of a step fail (previously a false `success=True`).
- LLM response parsing (shared across 6 OpenAI-compatible providers): clear
  errors on an empty `choices`, `refusal`, or truncated JSON instead of a bare
  `KeyError`/`JSONDecodeError`.
- Agent JSON parsing: non-dict results are rejected; one corrective retry on an
  invalid orchestrator response; crashes in plan logging eliminated.
- Timeouts on all search tools and in the MCP bridge (previously a hung server
  blocked the pipeline forever).
- `delete(ids=[])` in VectorDB no longer wipes the entire collection — an empty
  list/filter is rejected with an error.
- Chunker: the `chars <= max_chars` contract is honoured for code and tables;
  separators (`. `) are no longer lost at boundaries; overlap is actually
  applied.
- `Chain.run()` is thread-safe — concurrent runs (the Pool pattern) no longer
  interleave history; the stale `accumulated` after an error is eliminated.
- `from yait_aichain.tools import *` no longer fails — 7 non-existent names were
  removed from `__all__`.

### Changed
- The retry policy now covers POST requests (on 429/503 — safe pre-inference
  statuses); previously retry on 429/5xx fired for no generative call at all.
- Anthropic: with `reasoning=medium/high`, `max_tokens` is automatically raised
  above `budget_tokens` (previously a guaranteed HTTP 400).
- OpenAI: gpt-5 and the o-series no longer receive `temperature`/`top_p`, which
  these models reject; `reasoning` for gpt-5 now reaches the Responses API.
- Chroma: migrated from the remote REST API v1 to v2 (tenant/database, env-vars
  `CHROMA_TENANT`/`CHROMA_DATABASE`, idempotent `get_or_create`).
- Qdrant: string IDs (`"doc_1"`) are accepted — mapped to UUIDv5 while
  preserving the original; previously an HTTP 400.
- VectorDB `upsert` is batched (50 records at a time) — large uploads no longer
  fail on provider limits.
- `Skill` templating survives literal curly braces in prompts (for example,
  JSON samples); previously a `ValueError`.
- sttGoogle: sync/chunked is selected by audio duration rather than file size
  (previously a long low-bitrate file < 10 MB was guaranteed to fail).

### Security
- The Google API key is no longer passed in the query-string URL (it leaked
  into proxy/server logs) — moved to the `x-goog-api-key` header (client,
  model, embedder).

## [1.0.2] — 2026-05-17
### Added
- ...

## [1.0.0]
### Added
- `Skill` — call any LLM through a single interface
- `Chain` — sequential composition of steps
- `Pool` — parallel execution of steps
- `Agent` — autonomous multi-step reasoning
- `Tool` / `MCPTools` — tools and MCP servers
- `VectorDB` — vector search and RAG
- `Reranker` — result reranking
- 8 providers: Anthropic, OpenAI, Google, xAI, Perplexity, Kimi, DeepSeek, Qwen
