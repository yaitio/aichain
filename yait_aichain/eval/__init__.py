"""Eval: run cases through several arms, several times, and believe the result.

:class:`Eval` runs arms × cases × trials into a durable JSON-lines ledger;
:class:`Report` counts — mean, ``pass^k`` and ``flips`` — from those records
or from any foreign ones via ``Report.of()``, with controls, validity guards
and paired tests. The two halves do not depend on each other.
See docs/control/eval.md.
"""

from ._records import Case, Record, adapt, read, write
from ._report import Report, ci, se
from ._eval import Eval
from ._scorers import (
    Scorer, abstain, all_of, contains, exact, judge, normalise, numeric,
    pairwise, regex,
)

__all__ = [
    "Eval", "Case", "Record", "Report",
    "exact", "contains", "regex", "numeric", "judge", "pairwise",
    "abstain", "all_of",
    "adapt", "read", "write", "normalise", "se", "ci", "Scorer",
]
