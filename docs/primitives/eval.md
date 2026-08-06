# Eval

Run the same cases through several arms, several times, and be able to believe
the result.

```python
from yait_aichain.eval import Eval, Case, contains

ev = Eval(
    cases  = [Case("q1", input="capital of Norway?", expect="Oslo")],
    arms   = {"luna": lambda c: ask(c, "gpt-5.6-luna"),
              "oss":  lambda c: ask(c, "private/gpt-oss-20b")},
    score  = contains(),
    trials = 3,
)

ev.smoke()                  # two cases, every arm, before the bill
report = ev.run()
print(report.table())
```

```
arm                       mean   pass^k   flips  cases  err         $
─────────────────────────────────────────────────────────────────────
oss                      0.500    0.500   0.000      4    0    0.0031
luna                     0.500    0.000   1.000      4    0    0.1840
```

Two arms, identical accuracy, and one of them is useless. That is the whole
reason there are three columns.

---

## The three columns

| column | question it answers |
|---|---|
| `mean` | share of **attempts** that passed — accuracy |
| `pass^k` | share of **cases** passed on *every* trial — reliability |
| `flips` | share of cases answered **differently** on different trials |

They disagree, and the disagreement is usually the finding. On one regression
run the plain reference agent led on mean reward (0.467) and trailed on
`Pass^3` (0.300) behind a framework at 0.433 / 0.400 — reporting only the mean
would have ranked them backwards on the question actually being asked.

`flips` shows what neither of the others can. Two arms can reach the same mean
with one solving a fixed half of the set every time and the other solving a
different half each run. Only the second is unreliable. Read it beside the
mean and never alone: zero flips also describes something reproducibly wrong.

---

## Arms

An arm is a **named callable** taking a `Case`. Anything that can be called
fits — a `Skill`, a `Chain`, an `Agent`, a raw HTTP call, another framework.

```python
def ask(case, model):
    skill = Skill(Model(model),
                  {"messages": [{"role": "user", "parts": [case.input]}]})
    answer = skill.run()
    return {"output": answer,
            "cost":   skill.last_usage.cost,
            "tokens": skill.last_usage.total_tokens}
```

Return the answer, or a dict with `output` / `cost` / `tokens` when the arm
knows what it spent. Extra keys are kept in `meta`.

Keep everything except the thing under test identical between arms. A
paraphrased system prompt — a two-character difference — once cost a whole
study two thirds of its scores on a weak model.

---

## Scorers

```python
exact(fold=True)              # equals expect, whitespace collapsed
contains(all_of=True)         # every expected string present; partial score
regex()                       # expect is a pattern
numeric(tolerance=0.5)        # first number in the answer
judge(model, rubric="...")    # an LLM decides
all_of(a, b)                  # every scorer must pass
```

Your own scorer is a function of `(case, output)` returning a bool, a number,
an `(ok, score)` pair, or a dict with `ok`.

`judge` is part of the instrument, not the system under test: use the **same
model and prompt for every arm**, and check it with controls before trusting
it. A verdict it cannot parse raises rather than defaulting to a pass —
guessing "pass" lifts every arm at once and reads as a good day.

---

## Controls — run these before believing anything

```python
oracle = Eval(cases, {"oracle": lambda c: c.expect}, score).run()
noise  = Eval(cases, {"noise":  lambda c: "unrelated text"}, score).run()

print(Report.controls(oracle, noise))
# {'oracle': 1.0, 'noise': 0.0, 'ok': True, 'why': ''}
```

Feed the harness known-correct answers and known-wrong ones. The oracle must
score near 1.0 and the noise near 0.0. If the oracle comes in at 0.7, the
**metric** is broken, not the system, and every comparison built on it is
biased in a direction nothing in the output will reveal.

Half an hour, and it is the only check that can tell you the numbers
themselves are worthless.

---

## Validity guards

```python
report.reject(max_empty=2, max_errors=5)
# {'arm-c': '7 errors'}
```

A cell with too many empty outputs or provider errors is **absent from every
figure**, not scored as zero. A zero is a measurement; a rejected cell is the
absence of one, and printing it as zero invents a result.

---

## Comparing two arms

```python
report.paired("aichain", "langgraph")
# {'only_a': 8, 'only_b': 0, 'tied': 10, 'discordant': 8, 'p': 0.0078, 'enough': True}
```

Unpaired comparison wastes a small set: at 30 cases the standard error of a
proportion is ≈0.09, so nothing under ~0.18 separates. Pairing throws away the
cases both arms agree on — they carry no information about the difference —
and counts only the discordant ones. `enough` is false below ~6 discordant
pairs, where no honest test can reject anything.

`se(p, n)` is available separately; print it next to any proportion.

---

## Groups

```python
Case("t17", group="contested")
report.by_group()
```

Aggregate hides stratified effects. A regression set split into
always-solved / always-failed / contested had its entire result in how the
groups behaved differently — pooled, it showed nothing.

---

## The ledger

Every attempt is written to `./eval-runs/<name>.jsonl` as it happens, and on
start the file is read back so nothing already finished is re-run. An
interrupted run costs nothing to continue; a crash at attempt 400 does not
throw away 399.

The default output path is durable on purpose. A regression run of 450
simulations once existed only in a scratch directory, was reported as a table,
and is gone — writing somewhere that outlives the process has to be the
default, because the copy step is the step that gets forgotten.

Every number in every table is recounted from those records. Nothing reads a
summary produced by whatever ran the work: that shortcut once reported 0.878
where the raw data held 0.798.

---

## Reporting someone else's run

```python
from yait_aichain.eval import Report

report = Report.of(rows, arm="agent", case="task_id", ok="success")
print(report.by_group())
```

`Report` will count records it did not produce. Benchmarks that own their own
loop — and most do — still get `Pass^k`, stratified tables, paired tests and
validity guards without being rewritten. Unmapped keys land in `meta`, so
nothing is dropped on the way in.

---

## Concurrency

`run(concurrency=N)` runs attempts in threads, and the default is 1 on
purpose. An arm holding a single HTTP client, or an event loop bound to the
thread that created it, produces **empty results under concurrency and reports
no error at all**. That failure cost a whole benchmark cell before it was
understood. Raise it only when every arm is safe to call from several threads.
