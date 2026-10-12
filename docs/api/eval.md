# API: Eval

Generated from the code's signatures and docstrings. What it is for and how to use it: [the guide](../control/eval.md).

<!-- g:api -->
[`Eval`](#eval) · [`Case`](#case) · [`Record`](#record) · [`Report`](#report) · [`exact`](#exact) · [`contains`](#contains) · [`regex`](#regex) · [`numeric`](#numeric) · [`judge`](#judge) · [`pairwise`](#pairwise) · [`abstain`](#abstain) · [`all_of`](#all_of) · [`adapt`](#adapt) · [`read`](#read) · [`write`](#write) · [`normalise`](#normalise) · [`se`](#se)

---

## `Eval`

```python
Eval(
    cases: Iterable[Case],
    arms: dict[str, Callable[[Case], Any]],
    score: Scorer,
    *,
    trials: int = 1,
    out: str | Path | None = None,
    name: str = 'eval',
    verbose: bool = True,
)
```

The same cases through several arms, several times each, into a ledger.

Results are written as they happen to a durable JSON-lines file, a
finished `(arm, case, trial)` is never run twice, and `smoke()` tries
a couple of cases first. `run()` returns a `Report`.

**Parameters**

- `cases` (*iterable of Case*) — The work, each with a unique, stable `id`.
- `arms` (*dict*) — Name → a callable taking a `Case`. It returns the answer, or
  `{"output": …, "cost": …, "tokens": …}` when it knows what it spent.
  A Skill, a Chain, an Agent or someone else's code all fit.
- `score` (*callable*) — A scorer — `exact()`, `contains()`, `judge(model)`, … — or any
  function of `(case, output)` returning a bool, a number or a dict.
- `trials` (*int, optional*, default `1`) — Attempts per arm and case; `Pass^k` needs more than one.
- `out` (*str | Path | None, optional*, default `None`) — The ledger; `./eval-runs/<name>.jsonl` by default, never a scratch
  directory.
- `name` (*str, optional*, default `'eval'`) — The run's name, and the ledger's.
- `verbose` (*bool, optional*, default `True`) — Log each attempt as it finishes, to the `yait_aichain.eval` logger
  with a console handler installed.

### `Eval.smoke()`

```python
Eval.smoke(n: int = 2) -> Report
```

Run the first `n` cases through every arm, once, and report.

Nothing is written to the main ledger — a smoke run is a check on the
wiring, not data, and mixing the two means the real run resumes over
results taken under different conditions.

**Parameters**

- `n` (*int, optional*, default `2`) — How many cases, from the start.

### `Eval.run()`

```python
Eval.run(*, concurrency: int = 1) -> Report
```

Execute every `(arm, case, trial)` not already on disk.

`concurrency` above 1 runs attempts in threads. Use it only when
every arm is safe to call from several threads at once: an arm holding
one HTTP client, or an event loop bound to the thread that made it,
produces empty results under concurrency and reports no error at all —
that failure cost a whole benchmark cell before it was understood. The
default is 1 for that reason.

**Parameters**

- `concurrency` (*int, optional*, default `1`) — Attempts in flight at once.

---

## `Case`

```python
Case(
    id: str,
    input: Any = None,
    expect: Any = None,
    group: str = '',
    meta: dict = <factory>,
)
```

One unit of work, and what a correct answer to it looks like.

`id` is the join key across arms and trials, so it has to be stable
across runs — a position in a list is not an id, because inserting a case
renumbers every one after it and a comparison against yesterday silently
pairs the wrong rows.

`group` is optional and exists for stratified reporting: the A+
regression set splits its tasks into always-solved / always-failed /
contested, and the interesting result lived entirely in how the groups
behaved differently. A benchmark reported only in aggregate cannot show
that.

**Attributes**

- `id` (*str*) — The join key across arms and trials; stable from run to run.
- `input` (*any*, default `None`) — What an arm is given.
- `expect` (*any*, default `None`) — What a correct answer looks like, for the scorer.
- `group` (*str*, default `''`) — A stratum to report apart — `by_group()` reads it.
- `meta` (*dict*, default `<factory>`) — Anything else an arm or a scorer needs.

---

## `Record`

```python
Record(
    arm: str,
    case: str,
    trial: int,
    ok: bool = False,
    score: float = 0.0,
    output: Any = None,
    error: str = '',
    group: str = '',
    seconds: float = 0.0,
    tokens: int = 0,
    cost: float = 0.0,
    meta: dict = <factory>,
)
```

One arm's attempt at one case, on one trial.

`ok` is the scorer's verdict and `score` its magnitude, kept apart on
purpose. All-or-nothing throws away the difference between three of four
and none of four; a mean throws away whether anything was ever fully
right. Both are cheap, so both are kept.

`error` being set does not mean `ok` is False by definition — a
scenario may recover — but it is what the validity guards count, and a
cell with too many of them is rejected rather than reported.

**Attributes**

- `arm` (*str*) — The arm that made the attempt.
- `case` (*str*) — The case's id.
- `trial` (*int*) — Which attempt, from 1.
- `ok` (*bool*, default `False`) — The scorer's verdict.
- `score` (*float*, default `0.0`) — Its magnitude, 0 to 1.
- `output` (*any*, default `None`) — What the arm returned.
- `error` (*str*, default `''`) — The exception, if the arm raised.
- `group` (*str*, default `''`) — The case's group.
- `seconds` (*float*, default `0.0`) — How long the attempt took.
- `tokens` (*int*, default `0`) — Tokens it reported.
- `cost` (*float*, default `0.0`) — Dollars it reported.
- `meta` (*dict*, default `<factory>`) — The scorer's reasons and anything else recorded with it.

---

## `Report`

```python
Report(records: Iterable[Record])
```

Read-only view over a set of records, grouped by arm and case.

**Parameters**

- `records` (*iterable of Record*) — The attempts, usually read back from a ledger.

### `Report.of()`

```python
Report.of(source, **keys) -> Report
```

Build from records, a JSON-lines path, or someone else's dicts.

`Report.of("run.jsonl")` reads ours. `Report.of(rows, arm="agent",
case="task_id", ok="success")` adapts a foreign shape — which is how
a τ²-bench run or a retrieval bench gets reported here without being
run here.

**Parameters**

- `source` (*str | Path | list*) — A ledger path, a list of `Record`, or a list of foreign dicts.
- `**keys` (*str*) — For foreign dicts, which of their keys mean `arm`, `case`, `ok`, …

### `Report.reject()`

```python
Report.reject(
    *,
    max_empty: int = 2,
    max_errors: int = 5,
    max_abstained: float = 0.2,
) -> dict[str, str]
```

Mark cells whose data is not worth reporting, and say why.

The thresholds are the ones the τ²-bench runners used, where they
fired three times: twice on exhausted credits and once on an adapter
defect that produced empty dialogues under concurrency. A guard that
never fires is not evidence of clean data; these fired, and each time
the alternative was publishing a number that meant nothing.

**Parameters**

- `max_empty` (*int, optional*, default `2`) — Attempts with no output a cell may have.
- `max_errors` (*int, optional*, default `5`) — Attempts that raised a cell may have.
- `max_abstained` (*float, optional*, default `0.2`) — The share of attempts a judge could not read; past it the cell is
  rejected rather than reported on a shrunken denominator.

### `Report.abstentions()`

```python
Report.abstentions(arm: str, group: str = '') -> int
```

How many rows the judge declined to score for *arm*.

**Parameters**

- `arm` (*str*) — The arm.
- `group` (*str, optional*, default `''`) — Only this group.

### `Report.mean()`

```python
Report.mean(arm: str, group: str = '') -> float
```

Share of *attempts* that passed — pooled over cases, not averaged.

**Parameters**

- `arm` (*str*) — The arm.
- `group` (*str, optional*, default `''`) — Only this group.

### `Report.pass_k()`

```python
Report.pass_k(arm: str, k: int | None = None, group: str = '') -> float
```

Share of *cases* an arm solved on **every** one of k trials.

`k` defaults to the number of trials actually present. A case with
fewer than k trials is skipped rather than counted as a failure —
being under-sampled is not the same as being wrong, and folding the
two together makes an interrupted run look like a bad one.

**Parameters**

- `arm` (*str*) — The arm.
- `k` (*int | None, optional*, default `None`) — Trials that must all pass; the number of trials by default.
- `group` (*str, optional*, default `''`) — Only this group.

### `Report.flips()`

```python
Report.flips(arm: str, group: str = '') -> float
```

Share of cases the arm did **not** answer the same way every time.

This is the shape that a mean and a `Pass^k` both hide. Two arms can
land on the same mean with one solving a fixed half of the set every
time and the other solving a different half each run; only the second
is unreliable, and only this number says so.

Zero means perfectly reproducible — including reproducibly wrong,
which is why it is read next to the mean and never on its own.

**Parameters**

- `arm` (*str*) — The arm.
- `group` (*str, optional*, default `''`) — Only this group.

### `Report.paired()`

```python
Report.paired(a: str, b: str, group: str = '') -> dict
```

Compare two arms on the cases they both attempted.

Unpaired comparison wastes most of a small set: at 30 cases the
standard error of a proportion is ≈0.09, so nothing under ~0.18
separates. Pairing discards the cases both arms agree on — which carry
no information about the difference — and counts only the discordant
ones, where the whole signal lives.

`only_a` and `only_b` are those counts. Under the null they are
equal, so `p` is an exact two-sided binomial test on them; with fewer
than ~6 discordant pairs no honest test can reject anything, and the
function says so in `enough`.

**Parameters**

- `a` (*str*) — One arm.
- `b` (*str*) — The other.
- `group` (*str, optional*, default `''`) — Only this group.

### `Report.controls()`

```python
Report.controls(
    oracle: Report,
    noise: Report,
    *,
    arm: str = '',
    floor: float = 0.95,
    ceiling: float = 0.05,
) -> dict
```

Check the instrument before believing it.

`oracle` is the harness fed known-correct answers and `noise` the
same harness fed known-wrong ones. The oracle must score near 1.0 and
the noise near 0.0; anything else means the metric is broken rather
than the system, and every comparison built on it is biased in a
direction nobody can see.

Half an hour to run, and it is the only check here that can tell you
the numbers themselves are worthless.

**Parameters**

- `oracle` (*Report*) — The harness fed known-correct answers.
- `noise` (*Report*) — The harness fed answers that do nothing.
- `arm` (*str, optional*, default `''`) — Which arm of each to read; the only one by default.
- `floor` (*float, optional*, default `0.95`) — The least the oracle must score.
- `ceiling` (*float, optional*, default `0.05`) — The most the noise may score.

### `Report.table()`

```python
Report.table(group: str = '', k: int | None = None) -> str
```

One row per arm: accuracy, reliability, and what it cost.

**Parameters**

- `group` (*str, optional*, default `''`) — Only this group.
- `k` (*int | None, optional*, default `None`) — The `k` of `Pass^k`; the number of trials by default.

### `Report.by_group()`

```python
Report.by_group(k: int | None = None) -> str
```

The same table, once per group. Aggregate hides stratified effects.

**Parameters**

- `k` (*int | None, optional*, default `None`) — The `k` of `Pass^k`.

---

## `exact()`

```python
exact(*, fold: bool = True) -> Scorer
```

The answer equals `case.expect`, whitespace collapsed.

**Parameters**

- `fold` (*bool, optional*, default `True`) — Ignore case.

---

## `contains()`

```python
contains(*, fold: bool = True, all_of: bool = True) -> Scorer
```

Every string in `case.expect` appears in the answer.

`expect` may be one string or a list. With `all_of=False` any single
hit passes, and the partial score is the fraction found — the difference
between three of four and none of four is the whole reason a benchmark
discriminates, and an all-or-nothing verdict throws it away.

**Parameters**

- `fold` (*bool, optional*, default `True`) — Ignore case.
- `all_of` (*bool, optional*, default `True`) — Every expected string must appear; `False` — any one.

---

## `regex()`

```python
regex(*, flags: int = 0) -> Scorer
```

`case.expect` is a pattern the answer must match somewhere.

**Parameters**

- `flags` (*int, optional*, default `0`) — `re` flags.

---

## `numeric()`

```python
numeric(*, tolerance: float = 0.0) -> Scorer
```

The first number in the answer equals `case.expect`, within tolerance.

**Parameters**

- `tolerance` (*float, optional*, default `0.0`) — The largest difference still counted equal.

---

## `judge()`

```python
judge(
    model,
    *,
    rubric: str = 'The answer is correct if it states the same fact as the reference. Wording may differ.',
    prompt: str = 'You are grading one answer against a reference. Reply with exactly one word on the first line — PASS or FAIL — and nothing else on that line.\n\n{rubric}\n\nQUESTION:\n{question}\n\nREFERENCE:\n{expect}\n\nANSWER:\n{answer}',
) -> Scorer
```

An LLM decides. Use when the answer is prose and no rule can check it.

The judge is part of the instrument, not part of the system under test, so
two things follow. It must be **the same model and prompt across every
arm** — a judge swapped mid-run makes the arms incomparable in a way that
leaves no trace in the numbers. And it must be **checked before it is
trusted**: run the oracle and noise controls from `_report`, which
feed it known-correct and known-wrong answers. A judge that does not score
the oracle near 1.0 is measuring something other than correctness.

An unparseable verdict raises. That is deliberate: the alternative is to
guess, and guessing "pass" raises every arm at once, which reads as a good
result rather than as a broken judge.

**Parameters**

- `model` (*Model*) — The judge.
- `rubric` (*str, optional*, default a default) — What correct means here.
- `prompt` (*str, optional*, default a default) — The whole judging prompt, with `{rubric}`, `{question}`, `{expect}` and `{answer}`.

---

## `pairwise()`

```python
pairwise(
    model,
    *,
    rubric: str = 'Which answer better fulfils the question? Consider accuracy first, then completeness, then concision.',
    champion: str = 'the reference',
) -> Scorer
```

A judge compares the candidate with the champion, **in both orderings**.

LLM judges have a documented preference for whichever answer they read
first, and a single-ordering comparison measures that preference as much
as it measures quality. So the pair is put twice, A/B and B/A, and the
candidate wins only by winning both. A disagreement between the two
orderings *is* the position bias showing itself, and it resolves **to the
champion**: the burden is on the challenger, which is what keeps a
best-so-far from drifting on noise.

The champion is `case.expect` — the same field `judge` reads as
the reference.

`extra` carries what each ordering said, so a run can report how often
they disagreed. That number is the judge's own reliability, measured for
free while grading, and a pair of orderings that disagree half the time
means the comparison is a coin toss wearing a rubric.

**Parameters**

- `model` (*Model*) — The judge.
- `rubric` (*str, optional*, default a default) — What better means here.
- `champion` (*str, optional*, default `'the reference'`) — What the candidate is compared with, as named to the judge.

---

## `abstain()`

```python
abstain(why: str) -> dict
```

A verdict of *no verdict* — the judge could not judge this one.

`ok` is False so nothing downstream that ignores the flag can read an
abstention as a pass; `abstained` is what `Report`
reads, and it removes the row from the denominator rather than counting it
as a failure. Those two are different claims and the difference is the
whole point: a judge that cannot read an answer has said nothing about it.

**Parameters**

- `why` (*str*) — Why no verdict was possible; counted in `Report.abstentions()`.

---

## `all_of()`

```python
all_of(*scorers) -> Scorer
```

Passes when every scorer passes; the score is the mean of theirs.

---

## `adapt()`

```python
adapt(rows: Iterable[dict], **keys) -> Iterator[Record]
```

Map someone else's rows onto `Record`.

`adapt(rows, arm="agent", case="task_id", ok="success")` — the keyword is
our field, the value is theirs. Anything unmapped keeps its own name and
lands in `meta`, so nothing is dropped on the way in.

This exists because the two benches that most need this package do not run
through it: τ²-bench owns its own loop, and the byheart retrieval bench
scores documents rather than answers. Both can still be reported here, and
reporting is where the mistakes were.

---

## `read()`

```python
read(path) -> list[Record]
```

Read back what `write` produced, skipping lines a crash left broken.

---

## `write()`

```python
write(records: Iterable[Record], path) -> int
```

Append records as JSON lines.

Append rather than rewrite: a run that dies half-way keeps what it had,
and a truncated write costs one line instead of the file. The first A+ run
produced 450 simulations and survives only as a table someone pasted into
a chat, because nothing was ever written to a path that outlived the
process.

---

## `normalise()`

```python
normalise(verdict: Any) -> tuple[bool, float, dict]
```

Whatever a scorer returned, as `(ok, score, extra)`.

Accepts a bool, a number, a `(ok, score)` pair, or a dict carrying at
least `ok`. Anything else is a programming error and says so — a scorer
that silently returns `None` would otherwise mark every case failed and
look like a bad model.

---

## `se()`

```python
se(p: float, n: int) -> float
```

Standard error of a proportion — print it next to the proportion.

At n=184 this is ≈0.035, so differences under 0.07 are not distinguishable;
at n=30 it is ≈0.09 and almost nothing is. A number without its error bar
invites a ranking the data cannot support.
<!-- /g:api -->
