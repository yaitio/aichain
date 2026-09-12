"""
Turning attempts into numbers, and refusing to when the attempts are not sound.

Everything here recounts :class:`~._records.Record` objects. Nothing carries a
tally forward, and nothing reads a summary line produced by whatever ran the
work — the one time that shortcut was taken, the parser reported 0.878 where
the simulations held 0.798, and it was caught only because the raw data had
been kept.

Three habits are built in rather than left to the reader:

**Ratios are pooled, and the counts stay beside them.** Averaging per-case
ratios rewards silence: a case where an arm reported nothing has no wrong
answers in it and scores a perfect precision, so an arm that stays quiet on
half the set can finish ahead of one that answers. Counts are summed across
cases first, then divided once.

**A cell that cannot be trusted is absent, not zero.** A zero is a
measurement; a rejected cell is the absence of one, and printing it as zero
invents a result. :meth:`Report.reject` marks them and every figure skips them.

**Reliability is reported apart from accuracy.** ``Pass^k`` and the mean answer
different questions, and the A+ run showed them disagreeing: the reference
agent led on mean reward and trailed on ``Pass^3``. :meth:`Report.flips` goes
further and reports the shape neither of them shows — how often an arm gave
different verdicts to the same case on different trials.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

from ._records import Record


class Report:
    """Read-only view over a set of records, grouped by arm and case."""

    def __init__(self, records: "Iterable[Record]") -> None:
        self.records = [r for r in records]
        self._rejected: "dict[str, str]" = {}

    # ── construction ─────────────────────────────────────────────────────
    @classmethod
    def of(cls, source, **keys) -> "Report":
        """
        Build from records, a JSON-lines path, or someone else's dicts.

        ``Report.of("run.jsonl")`` reads ours. ``Report.of(rows, arm="agent",
        case="task_id", ok="success")`` adapts a foreign shape — which is how
        a τ²-bench run or a retrieval bench gets reported here without being
        run here.
        """
        from ._records import read, adapt
        if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__"):
            return cls(read(source))
        source = list(source)
        if source and isinstance(source[0], Record):
            return cls(source)
        return cls(adapt(source, **keys))

    # ── validity ─────────────────────────────────────────────────────────
    def reject(self, *, max_empty: int = 2, max_errors: int = 5,
               max_abstained: float = 0.2) -> "dict[str, str]":
        """
        Mark cells whose data is not worth reporting, and say why.

        The thresholds are the ones the τ²-bench runners used, where they
        fired three times: twice on exhausted credits and once on an adapter
        defect that produced empty dialogues under concurrency. A guard that
        never fires is not evidence of clean data; these fired, and each time
        the alternative was publishing a number that meant nothing.
        """
        for arm, rows in self._by_arm().items():
            empty  = sum(1 for r in rows if not r.error and _is_empty(r.output))
            errors = sum(1 for r in rows if r.error)
            silent = sum(1 for r in rows if self._abstained(r))
            if empty > max_empty:
                self._rejected[arm] = f"{empty} empty outputs"
            elif errors > max_errors:
                self._rejected[arm] = f"{errors} errors"
            # A judge that could not read most of an arm has not measured it.
            # Excluding those rows keeps the surviving number honest; past
            # this share there is no surviving number worth printing.
            elif rows and silent > max_abstained * len(rows):
                self._rejected[arm] = (f"{silent} of {len(rows)} unscored — "
                                       "the judge could not read them")
        return dict(self._rejected)

    @property
    def rejected(self) -> "dict[str, str]":
        return dict(self._rejected)

    def _live(self) -> "list[Record]":
        return [r for r in self.records if r.arm not in self._rejected]

    def _by_arm(self) -> "dict[str, list[Record]]":
        out: "dict[str, list[Record]]" = defaultdict(list)
        for r in self.records:
            out[r.arm].append(r)
        return dict(out)

    def arms(self) -> "list[str]":
        return sorted({r.arm for r in self._live()})

    def groups(self) -> "list[str]":
        return sorted({r.group for r in self._live() if r.group})

    # ── the numbers ──────────────────────────────────────────────────────
    @staticmethod
    def _abstained(record: Record) -> bool:
        """The judge said nothing about this row, so nothing may be counted.

        Excluded from the denominator rather than scored zero, because those
        are different claims: "wrong" and "unreadable by the instrument" fold
        into one number only if you are willing to blame the arm for the
        judge. The exclusions are counted and printed — a denominator that
        quietly shrinks is its own defect, and the arm a judge could not read
        twenty times has an instrument problem, not a 0.95.
        """
        return bool((record.meta or {}).get("abstained"))

    def abstentions(self, arm: str, group: str = "") -> int:
        """How many rows the judge declined to score for *arm*."""
        return sum(1 for r in self._live()
                   if r.arm == arm and (not group or r.group == group)
                   and self._abstained(r))

    def _cases(self, arm: str, group: str = "") -> "dict[str, list[Record]]":
        out: "dict[str, list[Record]]" = defaultdict(list)
        for r in self._live():
            if r.arm == arm and (not group or r.group == group) \
                    and not self._abstained(r):
                out[r.case].append(r)
        return dict(out)

    def mean(self, arm: str, group: str = "") -> float:
        """Share of *attempts* that passed — pooled over cases, not averaged."""
        rows = [r for c in self._cases(arm, group).values() for r in c]
        return sum(1 for r in rows if r.ok) / len(rows) if rows else 0.0

    def pass_k(self, arm: str, k: "int | None" = None, group: str = "") -> float:
        """
        Share of *cases* an arm solved on **every** one of k trials.

        ``k`` defaults to the number of trials actually present. A case with
        fewer than k trials is skipped rather than counted as a failure —
        being under-sampled is not the same as being wrong, and folding the
        two together makes an interrupted run look like a bad one.
        """
        cases = self._cases(arm, group)
        if k is None:
            k = max((len(v) for v in cases.values()), default=0)
        eligible = [rows for rows in cases.values() if len(rows) >= k]
        if not eligible or k == 0:
            return 0.0
        solved = sum(1 for rows in eligible
                     if all(r.ok for r in sorted(rows, key=lambda r: r.trial)[:k]))
        return solved / len(eligible)

    def flips(self, arm: str, group: str = "") -> float:
        """
        Share of cases the arm did **not** answer the same way every time.

        This is the shape that a mean and a ``Pass^k`` both hide. Two arms can
        land on the same mean with one solving a fixed half of the set every
        time and the other solving a different half each run; only the second
        is unreliable, and only this number says so.

        Zero means perfectly reproducible — including reproducibly wrong,
        which is why it is read next to the mean and never on its own.
        """
        cases = [rows for rows in self._cases(arm, group).values() if len(rows) > 1]
        if not cases:
            return 0.0
        unstable = sum(1 for rows in cases
                       if len({bool(r.ok) for r in rows}) > 1)
        return unstable / len(cases)

    def spend(self, arm: str, group: str = "") -> "dict[str, float]":
        rows = [r for c in self._cases(arm, group).values() for r in c]
        return {"cost":    sum(r.cost   for r in rows),
                "tokens":  sum(r.tokens for r in rows),
                "seconds": sum(r.seconds for r in rows),
                "errors":  sum(1 for r in rows if r.error),
                "trials":  len(rows),
                "cases":   len(self._cases(arm, group))}

    # ── comparison ───────────────────────────────────────────────────────
    def paired(self, a: str, b: str, group: str = "") -> dict:
        """
        Compare two arms on the cases they both attempted.

        Unpaired comparison wastes most of a small set: at 30 cases the
        standard error of a proportion is ≈0.09, so nothing under ~0.18
        separates. Pairing discards the cases both arms agree on — which carry
        no information about the difference — and counts only the discordant
        ones, where the whole signal lives.

        ``only_a`` and ``only_b`` are those counts. Under the null they are
        equal, so ``p`` is an exact two-sided binomial test on them; with fewer
        than ~6 discordant pairs no honest test can reject anything, and the
        function says so in ``enough``.
        """
        wins_a = wins_b = same = 0
        for case, rows_a in self._cases(a, group).items():
            rows_b = self._cases(b, group).get(case)
            if not rows_b:
                continue
            sa = sum(1 for r in rows_a if r.ok) / len(rows_a)
            sb = sum(1 for r in rows_b if r.ok) / len(rows_b)
            if   sa > sb: wins_a += 1
            elif sb > sa: wins_b += 1
            else:         same   += 1
        n = wins_a + wins_b
        p = _binom_two_sided(min(wins_a, wins_b), n) if n else 1.0
        return {"only_a": wins_a, "only_b": wins_b, "tied": same,
                "discordant": n, "p": p, "enough": n >= 6}

    # ── controls ─────────────────────────────────────────────────────────
    @staticmethod
    def controls(oracle: "Report", noise: "Report", *,
                 arm: str = "", floor: float = 0.95,
                 ceiling: float = 0.05) -> dict:
        """
        Check the instrument before believing it.

        ``oracle`` is the harness fed known-correct answers and ``noise`` the
        same harness fed known-wrong ones. The oracle must score near 1.0 and
        the noise near 0.0; anything else means the metric is broken rather
        than the system, and every comparison built on it is biased in a
        direction nobody can see.

        Half an hour to run, and it is the only check here that can tell you
        the numbers themselves are worthless.
        """
        a = arm or (oracle.arms() or [""])[0]
        b = arm or (noise.arms()  or [""])[0]
        got_hi, got_lo = oracle.mean(a), noise.mean(b)
        return {"oracle": got_hi, "noise": got_lo,
                "ok": got_hi >= floor and got_lo <= ceiling,
                "why": ("" if got_hi >= floor and got_lo <= ceiling else
                        f"oracle {got_hi:.3f} (want ≥{floor}), "
                        f"noise {got_lo:.3f} (want ≤{ceiling})")}

    # ── output ───────────────────────────────────────────────────────────
    def table(self, group: str = "", k: "int | None" = None) -> str:
        """One row per arm: accuracy, reliability, and what it cost."""
        head = (f"{'arm':<22}{'mean':>8}{'pass^k':>9}{'flips':>8}"
                f"{'cases':>7}{'err':>5}{'n/j':>5}{'$':>10}")
        lines = [head, "─" * len(head)]
        for arm in sorted(self.arms(),
                          key=lambda a: -self.pass_k(a, k, group)):
            s = self.spend(arm, group)
            lines.append(
                f"{arm:<22}{self.mean(arm, group):>8.3f}"
                f"{self.pass_k(arm, k, group):>9.3f}"
                f"{self.flips(arm, group):>8.3f}"
                f"{s['cases']:>7.0f}{s['errors']:>5.0f}"
                f"{self.abstentions(arm, group):>5.0f}{s['cost']:>10.4f}")
        for arm, why in sorted(self._rejected.items()):
            lines.append(f"{arm:<22}{'rejected — ' + why:>42}")
        if self.groups() and not group:
            lines.append("")
            lines.append("(pooled over groups: " + ", ".join(self.groups())
                         + " — read them apart, the A+ run's whole result was "
                           "in how they differed)")
        return "\n".join(lines)

    def by_group(self, k: "int | None" = None) -> str:
        """The same table, once per group. Aggregate hides stratified effects."""
        out = []
        for g in self.groups() or [""]:
            out.append(f"[{g or 'all'}]")
            out.append(self.table(group=g, k=k))
            out.append("")
        return "\n".join(out)


# ── helpers ──────────────────────────────────────────────────────────────────

def _is_empty(output: Any) -> bool:
    if output is None:
        return True
    if isinstance(output, str):
        return not output.strip()
    if isinstance(output, (list, dict, tuple)):
        return len(output) == 0
    return False


def _binom_two_sided(smaller: int, n: int) -> float:
    """Exact two-sided binomial test at p=0.5 — the sign test on pairs."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(smaller + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def se(p: float, n: int) -> float:
    """
    Standard error of a proportion — print it next to the proportion.

    At n=184 this is ≈0.035, so differences under 0.07 are not distinguishable;
    at n=30 it is ≈0.09 and almost nothing is. A number without its error bar
    invites a ranking the data cannot support.
    """
    return math.sqrt(p * (1 - p) / n) if n > 0 else 0.0
