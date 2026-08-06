"""
Running the same cases through several arms, N times each.

    ev = Eval(
        cases  = [Case("q1", input="capital of Norway?", expect="Oslo")],
        arms   = {"luna": lambda c: ask(c, "gpt-5.6-luna"),
                  "oss":  lambda c: ask(c, "private/gpt-oss-20b")},
        score  = contains(),
        trials = 3,
    )
    report = ev.run()
    print(report.table())

An **arm** is a named callable taking a :class:`~._records.Case`. It may return
the answer, or a dict ``{"output": ..., "cost": ..., "tokens": ...}`` when it
knows what it spent. Anything that can be called fits — a ``Skill``, a
``Chain``, an ``Agent``, a raw HTTP call, someone else's framework.

Three behaviours are not optional, because each of them was learned by losing
something:

**Results are written to a durable path as they happen.** ``out`` defaults to
``./eval-runs/<name>.jsonl`` under the working directory, never a scratch
directory. A regression run of 450 simulations once existed only in a session
scratchpad; it was reported as a table, the scratchpad was deleted, and the
raw data is gone. Writing to a place that outlives the process has to be the
default, or the copy step is the step that gets forgotten.

**A finished attempt is never re-run.** The output file is the ledger: on
start it is read back and every ``(arm, case, trial)`` already in it is
skipped. An interrupted run costs nothing to continue, and a crash at attempt
400 does not throw away 399.

**Smoke before scale.** :meth:`smoke` runs a couple of cases through every arm
and stops on trouble. Two minutes against two hours, and it catches the
failures that look identical from the outside — a missing key, a model name in
the wrong dialect, an arm that returns nothing at all.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Iterable

from ._records import Case, Record, read, write
from ._report import Report
from ._scorers import Scorer, normalise


class Eval:

    def __init__(
        self,
        cases:   "Iterable[Case]",
        arms:    "dict[str, Callable[[Case], Any]]",
        score:   Scorer,
        *,
        trials:  int = 1,
        out:     "str | Path | None" = None,
        name:    str = "eval",
        verbose: bool = True,
    ) -> None:
        self.cases   = list(cases)
        self.arms    = dict(arms)
        self.score   = score
        self.trials  = trials
        self.name    = name
        self.verbose = verbose

        self.out = Path(out) if out else Path.cwd() / "eval-runs" / f"{name}.jsonl"
        self.out.parent.mkdir(parents=True, exist_ok=True)

        ids = [c.id for c in self.cases]
        if len(set(ids)) != len(ids):
            # Cases are joined by id across arms and trials. Two cases sharing
            # one id silently merge, and Pass^k then counts a case that was
            # never attempted k times as if it had been.
            raise ValueError("case ids must be unique; duplicates: "
                             + ", ".join(sorted({i for i in ids
                                                 if ids.count(i) > 1})))

    # ── running ──────────────────────────────────────────────────────────
    def smoke(self, n: int = 2) -> "Report":
        """
        Run the first ``n`` cases through every arm, once, and report.

        Nothing is written to the main ledger — a smoke run is a check on the
        wiring, not data, and mixing the two means the real run resumes over
        results taken under different conditions.
        """
        probe = Eval(self.cases[:n], self.arms, self.score, trials=1,
                     out=self.out.with_suffix(".smoke.jsonl"),
                     name=f"{self.name}-smoke", verbose=self.verbose)
        report = probe.run()
        if self.verbose:
            print(report.table())
            bad = [a for a in probe.arms if report.spend(a)["errors"]
                   or report.spend(a)["trials"] == 0]
            print("\nsmoke: " + ("all arms answered" if not bad else
                                 "TROUBLE in " + ", ".join(bad)))
        return report

    def run(self, *, concurrency: int = 1) -> "Report":
        """
        Execute every ``(arm, case, trial)`` not already on disk.

        ``concurrency`` above 1 runs attempts in threads. Use it only when
        every arm is safe to call from several threads at once: an arm holding
        one HTTP client, or an event loop bound to the thread that made it,
        produces empty results under concurrency and reports no error at all —
        that failure cost a whole benchmark cell before it was understood. The
        default is 1 for that reason.
        """
        done = {(r.arm, r.case, r.trial) for r in self._existing()}
        todo = [(arm, case, t)
                for t in range(1, self.trials + 1)
                for case in self.cases
                for arm in self.arms
                if (arm, case.id, t) not in done]

        if self.verbose:
            total = len(self.arms) * len(self.cases) * self.trials
            print(f"{self.name}: {len(todo)} of {total} attempts to run "
                  f"({len(done)} already on disk) → {self.out}")

        if concurrency > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for rec in pool.map(lambda job: self._attempt(*job), todo):
                    self._commit(rec)
        else:
            for job in todo:
                self._commit(self._attempt(*job))

        report = Report(self._existing())
        report.reject()
        return report

    # ── one attempt ──────────────────────────────────────────────────────
    def _attempt(self, arm: str, case: Case, trial: int) -> Record:
        rec = Record(arm=arm, case=case.id, trial=trial, group=case.group)
        started = time.monotonic()
        try:
            raw = self.arms[arm](case)
        except Exception as exc:
            # One attempt failing is data, not a reason to stop: the guards
            # decide later whether there were too many. Dying here would throw
            # away every attempt already made.
            rec.error = f"{type(exc).__name__}: {exc}"
            rec.seconds = time.monotonic() - started
            return rec

        if isinstance(raw, dict) and "output" in raw:
            rec.output = raw.get("output")
            rec.cost   = float(raw.get("cost") or 0.0)
            rec.tokens = int(raw.get("tokens") or 0)
            rec.meta   = {k: v for k, v in raw.items()
                          if k not in ("output", "cost", "tokens")}
        else:
            rec.output = raw

        rec.seconds = time.monotonic() - started
        try:
            rec.ok, rec.score, extra = normalise(self.score(case, rec.output))
            rec.meta.update(extra)
        except Exception as exc:
            # A scorer that cannot decide must not decide. Recording the
            # failure keeps it visible to the validity guards instead of
            # letting a broken judge quietly mark everything passed.
            rec.error = f"scorer: {type(exc).__name__}: {exc}"
        return rec

    def _commit(self, rec: Record) -> None:
        write([rec], self.out)
        if self.verbose:
            mark = "!" if rec.error else ("+" if rec.ok else "·")
            note = f"  {rec.error[:60]}" if rec.error else ""
            print(f"  {mark} {rec.arm:<18} {rec.case:<28} "
                  f"t{rec.trial} {rec.seconds:5.1f}s{note}")

    def _existing(self) -> "list[Record]":
        return read(self.out) if self.out.exists() else []
