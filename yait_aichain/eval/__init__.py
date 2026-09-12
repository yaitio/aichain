"""
Eval — running a scenario many ways and being able to believe the result.

    from yait_aichain.eval import Eval, Case, contains

    ev = Eval(
        cases  = [Case("q1", input="capital of Norway?", expect="Oslo")],
        arms   = {"luna": lambda c: ask(c, "gpt-5.6-luna"),
                  "oss":  lambda c: ask(c, "private/gpt-oss-20b")},
        score  = contains(),
        trials = 3,
    )
    ev.smoke()                       # two cases, every arm, before the bill
    report = ev.run()
    print(report.table())            # mean, pass^k, flips, cost

The package is two halves that do not depend on each other.

:class:`Eval` **runs** — arms × cases × trials, writing every attempt to a
durable JSON-lines ledger as it happens, skipping whatever is already there.

:class:`Report` **counts** — and it will count records it did not produce.
``Report.of(rows, arm=..., case=..., ok=...)`` adapts a foreign shape, which is
how a τ²-bench run or a retrieval bench gets ``Pass^k``, stratified tables,
paired tests and validity guards without being rewritten to run here.

What it measures, and why there are three columns rather than one:

``mean``    share of attempts that passed — accuracy
``pass^k``  share of cases passed on *every* trial — reliability
``flips``   share of cases answered differently on different trials — the
            shape the other two hide

They disagree, and the disagreement is the finding. On a regression run the
reference agent led on mean reward (0.467) and trailed on ``Pass^3`` (0.300)
behind a framework at 0.433 / 0.400. Reporting only the mean would have ranked
them backwards on the question that was actually being asked.

Before believing any of it, run the controls: :meth:`Report.controls` takes an
oracle arm fed known-correct answers and a noise arm fed known-wrong ones. The
oracle must score near 1.0 and the noise near 0.0, or the metric is broken
rather than the system, and every comparison built on it is biased in a
direction nothing in the output will reveal.
"""

from ._records import Case, Record, adapt, read, write          # noqa: F401
from ._report import Report, se                                 # noqa: F401
from ._eval import Eval                                         # noqa: F401
from ._scorers import (                                         # noqa: F401
    Scorer, abstain, all_of, contains, exact, judge, normalise, numeric,
    pairwise, regex,
)

__all__ = [
    "Eval", "Case", "Record", "Report",
    "exact", "contains", "regex", "numeric", "judge", "pairwise",
    "abstain", "all_of",
    "adapt", "read", "write", "normalise", "se", "Scorer",
]
