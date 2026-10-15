"""Turning attempts into numbers, and refusing to when they are not sound.

Everything is recounted from records. Ratios are pooled with the counts kept
beside them; a cell that cannot be trusted is absent, not zero
(:meth:`Report.reject`); reliability (``Pass^k``, :meth:`Report.flips`) is
reported apart from accuracy. See docs/control/eval.md.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any
from collections.abc import Iterable

from ._records import Record


class Report:
    """Read-only view over a set of records, grouped by arm and case.

    Parameters
    ----------
    records : iterable of Record
        The attempts, usually read back from a ledger.
    """

    def __init__(self, records: Iterable[Record]) -> None:
        # One attempt is one (arm, case, trial). A ledger written twice for the
        # same attempt — a retried run, two writers — holds both lines, and
        # counting both would weigh that case double; the later line wins.
        latest = {(r.arm, r.case, r.trial): r for r in records}
        self.records = list(latest.values())
        self._rejected: dict[str, str] = {}
        # Indexed once: every figure below reads an arm's scored rows by case,
        # and rescanning every record for each one made `paired` and `table`
        # quadratic in the size of the run.
        self._scored: dict[str, dict[str, list[Record]]] = defaultdict(lambda: defaultdict(list))
        for r in self.records:
            if not self._abstained(r):
                self._scored[r.arm][r.case].append(r)

    # ── construction ─────────────────────────────────────────────────────
    @classmethod
    def of(cls, source, **keys) -> Report:
        """
        Build from records, a JSON-lines path, or someone else's dicts.

        ``Report.of("run.jsonl")`` reads ours. ``Report.of(rows, arm="agent",
        case="task_id", ok="success")`` adapts a foreign shape — which is how
        a τ²-bench run or a retrieval bench gets reported here without being
        run here.

        Parameters
        ----------
        source : str | Path | list
            A ledger path, a list of ``Record``, or a list of foreign dicts.
        **keys : str
            For foreign dicts, which of their keys mean ``arm``, ``case``, ``ok``, …
        """
        from ._records import read, adapt
        if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__"):
            return cls(read(source))
        source = list(source)
        if source and isinstance(source[0], Record):
            return cls(source)
        return cls(adapt(source, **keys))

    # ── validity ─────────────────────────────────────────────────────────
    def reject(self, *, max_empty: float = 0.1, max_errors: float = 0.2,
               max_abstained: float = 0.2) -> dict[str, str]:
        """
        Mark cells whose data is not worth reporting, and say why. Thresholds are
        shares of the arm's attempts, decided afresh on each call.

        Parameters
        ----------
        max_empty : float, optional
            The share of attempts that may come back with no output.
        max_errors : float, optional
            The share of attempts that may raise.
        max_abstained : float, optional
            The share of attempts a judge could not read; past it the cell is rejected.
        """
        self._rejected = {}
        for arm, rows in self._by_arm().items():
            n = len(rows) or 1
            empty  = sum(1 for r in rows if not r.error and _is_empty(r.output))
            errors = sum(1 for r in rows if r.error)
            silent = sum(1 for r in rows if self._abstained(r))
            if empty > max_empty * n:
                self._rejected[arm] = f"{empty} of {len(rows)} empty outputs"
            elif errors > max_errors * n:
                self._rejected[arm] = f"{errors} of {len(rows)} errors"
            # A judge that could not read most of an arm has not measured it.
            elif silent > max_abstained * n:
                self._rejected[arm] = (f"{silent} of {len(rows)} unscored — "
                                       "the judge could not read them")
        return dict(self._rejected)

    @property
    def rejected(self) -> dict[str, str]:
        return dict(self._rejected)

    def _live(self) -> list[Record]:
        return [r for r in self.records if r.arm not in self._rejected]

    def _by_arm(self) -> dict[str, list[Record]]:
        out: dict[str, list[Record]] = defaultdict(list)
        for r in self.records:
            out[r.arm].append(r)
        return dict(out)

    def arms(self) -> list[str]:
        return sorted({r.arm for r in self._live()})

    def groups(self) -> list[str]:
        return sorted({r.group for r in self._live() if r.group})

    # ── the numbers ──────────────────────────────────────────────────────
    @staticmethod
    def _abstained(record: Record) -> bool:
        """The judge said nothing about this row: left out of the denominator,
        not scored zero ("wrong" and "unreadable" are different claims), and
        counted, since a denominator that quietly shrinks is its own defect."""
        return bool((record.meta or {}).get("abstained"))

    def abstentions(self, arm: str, group: str = "") -> int:
        """How many rows the judge declined to score for *arm*.

        Parameters
        ----------
        arm : str
            The arm.
        group : str, optional
            Only this group.
        """
        return sum(1 for r in self._live()
                   if r.arm == arm and (not group or r.group == group)
                   and self._abstained(r))

    def _cases(self, arm: str, group: str = "") -> dict[str, list[Record]]:
        if arm in self._rejected or arm not in self._scored:
            return {}
        by_case = self._scored[arm]
        if not group:
            return dict(by_case)
        return {c: kept for c, rows in by_case.items()
                if (kept := [r for r in rows if r.group == group])}

    def mean(self, arm: str, group: str = "") -> float:
        """Share of *attempts* that passed — pooled over cases, not averaged.

        Parameters
        ----------
        arm : str
            The arm.
        group : str, optional
            Only this group.
        """
        rows = [r for c in self._cases(arm, group).values() for r in c]
        return sum(1 for r in rows if r.ok) / len(rows) if rows else 0.0

    def trials(self) -> int:
        """The number of trials, one for the whole report: the trials most
        cases have, across every arm. Per arm, an interrupted arm had a
        smaller k and read as more reliable — Pass^2 ranked beside Pass^3."""
        counts = Counter(len({r.trial for r in rows})
                         for by_case in self._scored.values() for rows in by_case.values())
        return counts.most_common(1)[0][0] if counts else 0

    def pass_k(self, arm: str, k: int | None = None, group: str = "") -> float:
        """
        The chance a case is solved on **every** one of k trials, by case.

        Estimated from all of a case's trials as C(c, k) / C(n, k), so Pass^1 is the
        mean; a case with fewer than k trials is left out, and no case at all is NaN.

        Parameters
        ----------
        arm : str
            The arm.
        k : int | None, optional
            Trials that must all pass; the report's number of trials by default.
        group : str, optional
            Only this group.
        """
        k = self.trials() if k is None else k
        cases = self._cases(arm, group)
        values = []
        for rows in cases.values():
            n = len({r.trial for r in rows})
            if k <= 0 or n < k:
                continue
            c = len({r.trial for r in rows if r.ok})
            values.append(math.comb(c, k) / math.comb(n, k))
        return sum(values) / len(values) if values else float("nan")

    def flips(self, arm: str, group: str = "") -> float:
        """
        Share of cases the arm did **not** answer the same way every time.

        Zero is perfectly reproducible — reproducibly wrong included — so read it
        beside the mean, never alone.

        Parameters
        ----------
        arm : str
            The arm.
        group : str, optional
            Only this group.
        """
        cases = [rows for rows in self._cases(arm, group).values() if len(rows) > 1]
        if not cases:
            return 0.0
        unstable = sum(1 for rows in cases
                       if len({bool(r.ok) for r in rows}) > 1)
        return unstable / len(cases)

    def spend(self, arm: str, group: str = "") -> dict[str, float]:
        """What the arm spent, over every attempt — abstained ones included,
        since they were paid for too. ``unpriced`` counts attempts with no
        cost, which ``cost`` leaves out.

        Parameters
        ----------
        arm : str
            The arm.
        group : str, optional
            Only this group.
        """
        cases = self._cases(arm, group)
        # Rejected arms too: a cell not worth reporting was still paid for.
        rows = [r for r in self.records if r.arm == arm and (not group or r.group == group)]
        return {"cost":    sum(r.cost for r in rows if r.cost is not None),
                "unpriced": sum(1 for r in rows if r.cost is None),
                "tokens":  sum(r.tokens for r in rows),
                "seconds": sum(r.seconds for r in rows),
                "errors":  sum(1 for r in rows if r.error),
                "trials":  len(rows),
                "cases":   len(cases)}

    # ── comparison ───────────────────────────────────────────────────────
    def paired(self, a: str, b: str, group: str = "") -> dict:
        """
        Compare two arms on the cases they both attempted, counting only the
        discordant ones: ``only_a``, ``only_b``, an exact two-sided binomial ``p``, and
        ``enough`` false below six discordant pairs.

        Parameters
        ----------
        a : str
            One arm.
        b : str
            The other.
        group : str, optional
            Only this group.
        """
        wins_a = wins_b = same = 0
        cases_b = self._cases(b, group)
        for case, rows_a in self._cases(a, group).items():
            rows_b = cases_b.get(case)
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
    def controls(oracle: Report, noise: Report, *,
                 arm: str = "", oracle_arm: str = "", noise_arm: str = "",
                 floor: float = 0.95, ceiling: float = 0.05) -> dict:
        """
        Check the instrument before believing it: the oracle must score at least
        *floor* and the noise at most *ceiling*, or the metric itself is broken.

        Parameters
        ----------
        oracle : Report
            The harness fed known-correct answers.
        noise : Report
            The harness fed answers that do nothing.
        arm : str, optional
            Which arm of each to read; the only one by default.
        oracle_arm : str, optional
            The oracle's arm, when the two reports name theirs differently.
        noise_arm : str, optional
            The noise report's arm, likewise.
        floor : float, optional
            The least the oracle must score.
        ceiling : float, optional
            The most the noise may score.
        """
        a = oracle_arm or arm or (oracle.arms() or [""])[0]
        b = noise_arm or arm or (noise.arms() or [""])[0]
        # No data is not a pass: a noise arm that was rejected, or misnamed,
        # scored 0.0 — exactly the "noise scores nothing" a passing control
        # needs — and the instrument was declared sound on nothing.
        for report, name, which in ((oracle, a, "oracle"), (noise, b, "noise")):
            if not report._cases(name):
                raise ValueError(f"controls: the {which} arm {name!r} has no scored attempts "
                                 f"(arms with data: {report.arms()}, rejected: {report.rejected})")
        got_hi, got_lo = oracle.mean(a), noise.mean(b)
        return {"oracle": got_hi, "noise": got_lo,
                "ok": got_hi >= floor and got_lo <= ceiling,
                "why": ("" if got_hi >= floor and got_lo <= ceiling else
                        f"oracle {got_hi:.3f} (want ≥{floor}), "
                        f"noise {got_lo:.3f} (want ≤{ceiling})")}

    # ── output ───────────────────────────────────────────────────────────
    def table(self, group: str = "", k: int | None = None) -> str:
        """One row per arm: accuracy, reliability, and what it cost.

        Parameters
        ----------
        group : str, optional
            Only this group.
        k : int | None, optional
            The ``k`` of ``Pass^k``; the number of trials by default.
        """
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

    def by_group(self, k: int | None = None) -> str:
        """The same table, once per group. Aggregate hides stratified effects.

        Parameters
        ----------
        k : int | None, optional
            The ``k`` of ``Pass^k``.
        """
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


def ci(passed: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    The Wilson interval for *passed* of *n* — which `se` cannot give at the
    edges: at 10 of 10 the standard error is 0, an interval of one point,
    where the data says only "probably above 0.72".

    Parameters
    ----------
    passed : int
        Successes.
    n : int
        Trials.
    z : float, optional
        The normal quantile; 1.96 for 95 %.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    p = passed / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))
