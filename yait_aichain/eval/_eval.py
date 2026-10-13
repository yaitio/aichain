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

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any
from collections.abc import Callable, Iterable

from .._console import console
from ._records import Case, Record, read, write
from ._report import Report
from ._scorers import Scorer, normalise

_log = logging.getLogger("yait_aichain.eval")


class Eval:
    """
    The same cases through several arms, several times each, into a ledger.

    Results are written as they happen to a durable JSON-lines file, a
    finished ``(arm, case, trial)`` is never run twice, and ``smoke()`` tries
    a couple of cases first. ``run()`` returns a :class:`Report`.

    Parameters
    ----------
    cases : iterable of Case
        The work, each with a unique, stable ``id``.
    arms : dict
        Name → a callable taking a ``Case``. It returns the answer, or
        ``{"output": …, "cost": …, "tokens": …}`` when it knows what it spent.
        A Skill, a Chain, an Agent or someone else's code all fit.
    score : callable
        A scorer — ``exact()``, ``contains()``, ``judge(model)``, … — or any
        function of ``(case, output)`` returning a bool, a number or a dict.
    trials : int, optional
        Attempts per arm and case; ``Pass^k`` needs more than one.
    out : str | Path | None, optional
        The ledger; ``./eval-runs/<name>.jsonl`` by default, never a scratch
        directory.
    name : str, optional
        The run's name, and the ledger's.
    verbose : bool, optional
        Log each attempt as it finishes, to the ``yait_aichain.eval`` logger
        with a console handler installed.
    """

    def __init__(
        self,
        cases:   Iterable[Case],
        arms:    dict[str, Callable[[Case], Any]],
        score:   Scorer,
        *,
        trials:  int = 1,
        out:     str | Path | None = None,
        name:    str = "eval",
        verbose: bool = True,
    ) -> None:
        if trials < 1:
            raise ValueError(f"trials is at least 1, not {trials}: a run of no "
                             "attempts reports nothing and reads as an empty success")
        self.cases   = list(cases)
        self.arms    = dict(arms)
        self.score   = score
        self.trials  = trials
        self.name    = name
        self.verbose = verbose
        if verbose:
            pass

        # Created on the first write, not here: an Eval built where the
        # working directory is read-only (a Lambda) must still be buildable.
        self.out = Path(out) if out else Path.cwd() / "eval-runs" / f"{name}.jsonl"

        # What a recorded attempt answered: this case's input and expectation,
        # scored by this scorer. A ledger reused under the same name with other
        # cases, or another scorer, is not resumed as if it were this run's.
        scorer = getattr(score, "__qualname__", type(score).__qualname__)
        self._fp = {c.id: hashlib.sha256(json.dumps([c.input, c.expect, scorer], default=repr,
                                                    sort_keys=True).encode()).hexdigest()[:16]
                    for c in self.cases}

        ids = [c.id for c in self.cases]
        if len(set(ids)) != len(ids):
            # Cases are joined by id across arms and trials. Two cases sharing
            # one id silently merge, and Pass^k then counts a case that was
            # never attempted k times as if it had been.
            raise ValueError("case ids must be unique; duplicates: "
                             + ", ".join(sorted({i for i in ids
                                                 if ids.count(i) > 1})))

    # ── running ──────────────────────────────────────────────────────────
    def smoke(self, n: int = 2) -> Report:
        """
        Run the first ``n`` cases through every arm, once, and report.

        Nothing is written to the main ledger — a smoke run is a check on the
        wiring, not data, and mixing the two means the real run resumes over
        results taken under different conditions.

        Parameters
        ----------
        n : int, optional
            How many cases, from the start.
        """
        smoke_file = self.out.with_suffix(".smoke.jsonl")
        if smoke_file.exists():
            smoke_file.unlink()            # a smoke run checks today's wiring, not yesterday's
        probe = Eval(self.cases[:n], self.arms, self.score, trials=1,
                     out=smoke_file, name=f"{self.name}-smoke", verbose=self.verbose)
        report = probe.run()
        if self.verbose:
            bad = [a for a in probe.arms if report.spend(a)["errors"]
                   or report.spend(a)["trials"] == 0]
            console().info("%s\n\nsmoke: %s", report.table(),
                      "all arms answered" if not bad else "TROUBLE in " + ", ".join(bad))
        return report

    def run(self, *, concurrency: int = 1) -> Report:
        """
        Execute every ``(arm, case, trial)`` not already on disk.

        ``concurrency`` above 1 runs attempts in threads. Use it only when
        every arm is safe to call from several threads at once: an arm holding
        one HTTP client, or an event loop bound to the thread that made it,
        produces empty results under concurrency and reports no error at all —
        that failure cost a whole benchmark cell before it was understood. The
        default is 1 for that reason.

        Parameters
        ----------
        concurrency : int, optional
            Attempts in flight at once.
        """
        # Done means answered under this configuration: an attempt that
        # failed — a 401, an empty balance — is run again once it is fixed,
        # where it counted as done and the resume made no calls; and one made
        # for other cases or another scorer, in a ledger of the same name, is
        # not this run's.
        done = {(r.arm, r.case, r.trial) for r in self._existing()
                if not r.error and (r.meta or {}).get("fp") in (None, self._fp.get(r.case))}
        todo = [(arm, case, t)
                for t in range(1, self.trials + 1)
                for case in self.cases
                for arm in self.arms
                if (arm, case.id, t) not in done]

        if self.verbose:
            total = len(self.arms) * len(self.cases) * self.trials
            console().info("%s: %d of %d attempts to run (%d already on disk) → %s",
                      self.name, len(todo), total, len(done), self.out)

        if concurrency > 1:
            # Written as each finishes, not in the order sent: one slow or
            # failed attempt held back the record of every one after it, and
            # an interrupted run lost attempts already paid for. Each in a
            # copy of this context, so a tenant and a budget reach the arms.
            import contextvars
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(contextvars.copy_context().run, self._attempt, *job)
                           for job in todo]
                for future in as_completed(futures):
                    self._commit(future.result())
        else:
            for job in todo:
                self._commit(self._attempt(*job))

        report = Report(self._existing())
        report.reject()
        return report

    # ── one attempt ──────────────────────────────────────────────────────
    def _attempt(self, arm: str, case: Case, trial: int) -> Record:
        rec = Record(arm=arm, case=case.id, trial=trial, group=case.group,
                     meta={"fp": self._fp[case.id]})
        started = time.monotonic()
        try:
            raw = self.arms[arm](case)
            self._unwrap(raw, rec)
        except Exception as exc:
            # One attempt failing is data, not a reason to stop: the guards
            # decide later whether there were too many. Dying here would throw
            # away every attempt already made.
            rec.error = f"{type(exc).__name__}: {exc}"
            rec.seconds = time.monotonic() - started
            return rec

        rec.seconds = time.monotonic() - started
        if rec.error:
            return rec
        try:
            rec.ok, rec.score, extra = normalise(self.score(case, rec.output))
            rec.meta.update(extra)
        except Exception as exc:
            # A scorer that cannot decide must not decide — and must not
            # decide against the arm either: a judge's provider failing is an
            # abstention, counted apart, not a wrong answer.
            rec.ok, rec.score = False, 0.0
            rec.meta.update(abstained=True, why=f"scorer: {type(exc).__name__}: {exc}")
        return rec

    @staticmethod
    def _unwrap(raw, rec: Record) -> None:
        """Read an arm's return into *rec*: a result object, an envelope, or
        the answer itself. Inside the attempt's try, so a malformed envelope
        is that attempt's error, not the run's."""
        # A primitive's result object — ChainResult, PoolResult, AgentResult
        # — records its output, and what it cost, rather than the object: the
        # ledger is JSON, and the object is not.
        if (not isinstance(raw, dict) and hasattr(raw, "output")
                and hasattr(raw, "success")):
            rec.output = raw.output
            cost = getattr(raw, "cost", None)
            rec.cost   = float(cost) if cost is not None else None
            rec.tokens = int(getattr(raw, "tokens_used", 0) or 0)
            rec.meta.update(success=bool(raw.success), run_error=getattr(raw, "error", None))
            # A run that failed is an error, not an answer of None: a dead
            # agent read as "all arms answered".
            if not raw.success:
                rec.error = f"run failed: {getattr(raw, 'error', None) or 'no reason given'}"
        elif isinstance(raw, dict) and "output" in raw:
            rec.output = raw.get("output")
            cost = raw.get("cost")
            rec.cost   = float(cost) if cost is not None else None
            rec.tokens = int(raw.get("tokens") or 0)
            rec.meta.update({k: v for k, v in raw.items() if k not in ("output", "cost", "tokens")})
        else:
            rec.output = raw

    def _commit(self, rec: Record) -> None:
        self.out.parent.mkdir(parents=True, exist_ok=True)
        write([rec], self.out)
        if self.verbose:
            mark = "!" if rec.error else ("+" if rec.ok else "·")
            note = f"  {rec.error[:60]}" if rec.error else ""
            console().info("  %s %-18s %-28s t%d %5.1fs%s",
                      mark, rec.arm, rec.case, rec.trial, rec.seconds, note)

    def _existing(self) -> list[Record]:
        return read(self.out) if self.out.exists() else []
