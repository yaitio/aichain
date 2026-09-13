# Cleanup plan — from the 2026-09-13 audit

Source: [audit-2026-09-13.md](audit-2026-09-13.md). This is the order of
work, with a measurable exit condition for every item, so "done" is a test
result and not an opinion. Nothing here adds a capability; every item makes
something that already exists true, checked, or generated.

The rule that runs through all of it: **where a page was written from data,
generate it from the data and hold it with a test.** The 2.6.1 hand-sweep
went stale on four pages within a week. `parameters.md` did not, because a
run produces it. Everything tabular goes the same way.

---

## Stage 0 — stop the bleeding (today, ~2 hours)

| # | item | exit condition |
|---|---|---|
| 0.1 | **`publish.yml` gates on the floor.** It already runs `pytest` before publishing — **on 3.12 only**, where the PEP 701 line is legal, which is exactly why eleven broken releases got through a gate that existed. Run the pre-publish test on **3.10**, and additionally wait on the `Tests` workflow for the same SHA. | A tag pushed on a commit that fails on 3.10 does not reach PyPI. Verified by pushing a deliberately failing tag to a branch. |
| 0.2 | **CI matrix: add 3.13 and 3.14.** Development is on 3.14; CI stops at 3.12. | `test.yml` matrix is `["3.10","3.11","3.12","3.13","3.14"]` and green. |
| 0.3 | **Branch protection on `main`**: `Tests` required. | GitHub settings; a push that fails `Tests` cannot land. (Owner action — needs repo admin.) |
| 0.4 | **Decide on yanking 2.7.0–2.17.0.** Eleven releases are unimportable on 3.10/3.11. | A decision recorded in `CHANGELOG.md` under 2.17.1 either way. (Owner action — PyPI web UI.) |
| 0.5 | **Land or shelve the `nudge` work in the tree.** It is implemented, tested, and found one real fix (the loop recorded the tool name as the intent, so `is_repeating` read ten different searches as one). | Either 2.18.0 released with the CI gate from 0.1 in place, or the diff moved to a branch and the tree clean. Recommendation: land it. |
| 0.6 | **Commit the audit and this plan.** | Both files on `main`, linked from `CLAUDE.md` and `docs/design/`. |

## Stage 1 — examples that run (half a day)

| # | item | exit condition |
|---|---|---|
| 1.1 | **Rewrite the three dead examples.** `16_debug.py`: `orchestrator=` → positional model. `20_observability.py`: same, and replace the "resumes after approval" claim with `approve=` (which is what actually gates now). `18_agent_external_trigger.py`: the cross-process pattern is a **Chain** capability; rewrite it as a Chain with a `Gate` step, keep the webhook story. | All 24 examples bind (1.2) and the three rewritten ones run offline with a scripted model where the original did. |
| 1.2 | **Put `examples/` under the docs-bind test.** `test_lightness_invariant.py` walks `docs/` and `README.md`; extend it to `examples/*.py` — every `Agent(`/`Skill(`/`Chain(`/`Pool(`/`Model(` call must bind to the real signature. | Test exists, passes, and fails when an example is broken (checked by breaking one). |
| 1.3 | **Generate `examples/README.md` from the scripts' docstrings.** The index described three dead examples as working because it is a hand-written table. | A script produces the table from each file's first docstring paragraph and required env vars; a test asserts the committed file equals the generated one. |
| 1.4 | **A "runs offline" smoke for every example that has a scripted model** (`20`, `22`, any that mock the transport). | Each such example is imported and executed in the suite under 5 s. |

## Stage 2 — documentation generated from data (two days)

Hand-fixing prose comes second; first, every table that has a source of
truth is produced from it, and a test holds the committed page equal to the
generated one — the `parameters.md` pattern.

| # | page | source of truth | exit condition |
|---|---|---|---|
| 2.1 | `docs/index.md` provider-by-modality table | `registry.providers(task=…)` | Generated block between markers; test holds it. Today: 3 image providers listed, 7 real. |
| 2.2 | `docs/reference/model-registry.md` | `registry.models(provider=…)` + per-provider TOML notes | Whole page generated. Today: 23 of 90 models missing, 3 providers missing, `dall-e` present. |
| 2.3 | `docs/reference/environment-variables.md` provider section | `PROVIDERS[*].provider.env_key` | Generated; today 4 of 11 keys missing. |
| 2.4 | `docs/tools-reference/index.md` **and one page per tool** | tool classes: `name`, `description`, `parameters` (options with types, enums, descriptions), `risk`, required env | 32 pages generated from schemas, not 7 hand-written ones. Today the search pages omit 12 declared options between them. Hand-written prose per tool, if any, lives in a `notes` block the generator preserves. |
| 2.5 | `docs/primitives/pool.md` and `chain.md` error-policy tables | `_errors_policy.POLICIES` and their docstrings | Generated block; today `pool.md` lacks `stop`. |
| 2.6 | README "N models from N providers" and the "12 providers" section | `registry` | Numbers generated at release time; a test compares. Today 88 vs 90. |
| 2.7 | `docs/primitives/models.md` option table | `_options.UNIVERSAL_OPTIONS` + `UNIVERSAL_FORMAT` + `extra`/`on_unsupported` | Generated; today 11 vocabulary entries are not on the page. |

Then the prose, by hand, once, with the bind test still holding the code:

| # | page | fix |
|---|---|---|
| 2.8 | `getting-started/concepts.md` | Rewrite the Agent section for the 2.x loop: one loop, decisions are tool calls or an answer, `stop_when`, no reflect phase, no `max_attempts`. Remove "waterfall stops if max attempts exceeded". |
| 2.9 | `docs/index.md` | Remove "Sectional generation" as a headline feature (it is a Chain docstring, not a feature); remove "Professional and Expert pipelines" unless it is documented somewhere a reader can find. |
| 2.10 | `agents/overview.md` | Modes table: `agile` is the default. |
| 2.11 | `agents/agent-as-chain-step.md` | Persistence list: drop `executors` and `result.memory`; list what `Chain.save` actually writes (`model`, `mode`, `stop_when`, `instructions`, `name`, tools). |
| 2.12 | `primitives/tools.md` | Add the governance section: `risk`, `PermissionPolicy`, `approve=`, `check_args` and option validation, `unknown_keys`. It is the page about tools and says nothing about any of them. |
| 2.13 | `primitives/eval.md` | Add `pairwise` and `abstain`, the `n/j` column, and the abstention rejection rule. |
| 2.14 | `docs/design/` | Link the design docs from `docs/index.md`. They are the best writing in the repository and unreachable. |

**Exit for Stage 2:** a `scripts/docs.py --check` that regenerates every
generated block and fails on any diff, run in CI beside the parameter
matrix.

## Stage 3 — cookbooks (one day, plus a decision)

| # | item | exit condition |
|---|---|---|
| 3.1 | **Six skeleton pages: link or delete.** `rag`→`cite`, `research-agent`→`survey`, `long-document`→`brief`, `image-pipeline`→`press` exist as ideas in `../aichain_cookbooks/rnd/`; `multi-provider-routing` and `translate-and-publish` map to nothing. | Four pages become one-paragraph pointers to the `rnd/` project with its status; two are deleted from the index. No page says "Skeleton". |
| 3.2 | **Apply the promotion rule.** `council` and `idntty-games` are promoted and have zero tests. | Each gains a test that fails when the library breaks it (a bind test at minimum, an offline run if scripted), or is demoted to `rnd/`. |
| 3.3 | **Fix `rnd/hAIve` and `rnd/resona`**, which use removed parameters. | Both import against 2.17.1. |
| 3.4 | **The cookbooks repository gets the same bind test** as `examples/`, over every `.py` in promoted projects. | Test in that repository, green. |

## Stage 4 — tests that measure what they claim (one day)

| # | item | exit condition |
|---|---|---|
| 4.1 | **Coverage in CI.** `pytest-cov`, a number in CI output, and a `COVERAGE.md` register: every deliberately uncovered region carries a reason (the elasticgraph pattern the plan already cites). | A coverage number exists; the register exists; a drop below the number fails CI. |
| 4.2 | **Live tests nightly.** A scheduled workflow with provider keys in secrets runs the 73 `Live` tests and posts a summary. Every "established by effect" claim becomes reproducible. | Workflow exists and has run green once. |
| 4.3 | **`Pool` and `Chain` tests in proportion.** 11 and 28 tests for the two primitives that compose everything. | Each covers: every error policy, budget lending, context propagation, beacons, save/load, hooks. Target ≥ 40 each. |
| 4.4 | **A test that CI is read.** Not a joke: `publish.yml` gated on `Tests` (0.1) is the mechanism; this item is the checklist entry that says it exists. | 0.1 done. |

## Stage 5 — prove goal 2: an agent can write it (two days)

| # | item | exit condition |
|---|---|---|
| 5.1 | **`llms.txt`** at the repository root: the five primitives, one example each, the real signatures, the three rules an agent gets wrong (`stop_when` not `max_steps`; `input=` or `prompt=`; `tool_result_turn` for every call). | File exists, under 300 lines, every example binds (covered by 1.2's pattern). |
| 5.2 | **`SKILL.md`** for Claude/Codex: how to build a Skill, a Chain, an Agent with tools, from `llms.txt`. | File exists. |
| 5.3 | **Twenty tasks with auto-checks**, run with a mid-tier model given only `SKILL.md` and `llms.txt`. Measured with `eval/` — `Pass^3`, three prompt-support levels per the measurement rules. | A number. If it is low, goal 2 is not met and the plan says so; if it is high, that is the headline for the README. |

### Stage 5 result, 2026-09-13

`gpt-5.4-mini`, given only `SKILL.md` and `llms.txt`: **`Pass^3` 0.60** on the
neutral level (0.65 supportive, 0.65 competing), 180 attempts, $0.66.
Analysis: [`evals/agent_builds/RESULTS.md`](../../evals/agent_builds/RESULTS.md).

**Goal 2 is partly met, and the gap is documentation, not the library.** 41 of
47 failures were things the page did not say; the rules it stated held — 35 of
36 attempts on the tasks that exercise them. The
largest single cause — `chain.run()` returning the last step's output where a
model expected a dict keyed by step names — is also an API question for the
owner: the docs now say it, but a return value that trips a careful reader in
six tasks out of twenty may be the wrong return value. Changing it is breaking.

Not yet measured, and named before anyone runs it: the rewritten `llms.txt`
on a **held-out** set of twenty tasks, same model, same levels. The threshold is
stated in advance — if neutral `Pass^3` does not rise above 0.60, the rewrite
did not help and the result is recorded as such.

## Stage 6 — the product decision (owner; not engineering)

`products/` has been empty since 17 May. Goal 1 is unmet until one of these
is written down:

| option | what it implies for the library |
|---|---|
| **A. Build the serverless product.** | Stage 5's `llms.txt` is the funnel's mouth; next library work is a Lambda handler template and a `deploy` example under `products/`, and the R1–R5 UI work has its consumer. |
| **B. No product; the library is a research instrument.** | Stop shipping UI-facing features. Invest in `eval/`, the compliance grader, the measurement discipline. Reframe the README. |
| **C. No product yet; keep the option.** | Nothing changes in code; the goal is restated as "nothing in the library prevents it" and the audit stops counting it as unmet. |

Exit: one line in `VISION.md` and one in `CLAUDE.md` naming the option.

---

## Explicitly not in this cleanup

These are real losses against peers and they are **new capability**, not
cleanup. They wait until the stages above are done:

- async (`Skill.arun` / `Agent.arun`)
- typed structured output (`output=PydanticModel`, `py.typed`)
- an OpenTelemetry exporter as a `Hook`
- PII masking (the deferred 2.4.0 guardrails)
- tool wire-name unification (`convertToMD` vs `vector_query`) — breaking

## Definition of clean

All true at once, and each is a test or a CI job rather than a claim:

- [ ] a tag on a red commit cannot publish
- [ ] CI covers 3.10 through 3.14 and is green
- [x] every fenced example in `docs/`, `README.md` and `examples/` binds
- [x] every generated docs block equals its regeneration (`scripts/docs.py --check`)
- [ ] no page contains the word "Skeleton"
- [ ] no promoted cookbook lacks a test
- [x] a coverage number is printed by CI and cannot fall
- [ ] the live suite has run green on a schedule at least once
- [x] an agent given `llms.txt` and `SKILL.md` has a measured `Pass^3` on twenty tasks
- [ ] `products/` is either non-empty or the goal is restated in writing

## Order and cost

Stage 0 today. Stages 1 → 2 → 3 → 4 in that order, about five working
days. Stage 5 in parallel with 3–4 once 2 is done, because it needs the
generated docs to be true. Stage 6 whenever the owner decides; it blocks
nothing above and everything after.
