"""
_budget — a ceiling in money, shared by everything under one run.

One object, passed down, decremented by whoever spends: a budget handed to a
`Chain` covers the agent inside it because both hold the object, not a copy of
a number each could spend in full. The money is counted in ``Skill``, the one
primitive that talks to a model. It bounds *do not begin another call*, not
*never exceed*: under concurrency the overshoot is up to one call per worker.
See docs/control/cost-and-budget.md.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar

from ._picklable import Picklable


class BudgetExceeded(Picklable, RuntimeError):
    """A run stopped because its money ceiling was reached.

    A raise rather than a partial result. Returning what was finished with a
    flag set is how a truncated run gets read as a complete one — the same
    silent-success shape this library has spent its 2.x line removing — and a
    caller who wants the partial output can catch this and read
    ``last_usage``.

    Parameters
    ----------
    spent : float
        Dollars spent when the call was refused.
    limit : float
        The ceiling.
    what : str, optional
        What was refused.
    """

    def __init__(self, spent: float, limit: float, what: str = "") -> None:
        super().__init__(
            f"budget exhausted{' on ' + what if what else ''}: "
            f"${spent:.6f} spent of ${limit:.6f}. Raise max_cost, or catch "
            "BudgetExceeded and read what completed.")
        self.spent = spent
        self.limit = limit
        self.what  = what


class Budget:
    """A ceiling in dollars, and what has been spent against it.

    Thread-safe: the sum is kept under a lock, because `Pool` runs its items
    concurrently. A call already in flight when the ceiling is reached still
    finishes, so concurrent work can overshoot by up to one call per worker.

    ::

        budget = Budget(0.50)
        Chain(steps=[...], max_cost=budget).run()
        budget.spent        # what it cost
        budget.remaining    # what is left

    A `Budget` passed to a primitive is shared, and spends down across every
    run that holds it. A number — ``max_cost=0.5`` — is a ceiling **per run**:
    each ``run()`` starts a fresh one, so an agent built once at module level
    in a warm Lambda does not refuse every request after the first expensive
    ones, and a resumed run starts from what the parked one had spent.

    Parameters
    ----------
    limit : float
        The ceiling, in dollars; must be positive.
    spent : float, optional
        What has already been spent against it.
    """

    def __init__(self, limit: float, *, spent: float = 0.0) -> None:
        import math
        if isinstance(limit, bool) or not isinstance(limit, (int, float)) \
                or not math.isfinite(limit) or limit <= 0:
            # NaN compares false with everything, so it would never be
            # exhausted; `True` would be a dollar.
            raise ValueError(f"a budget is a finite, positive number of dollars; got {limit!r}")
        self.limit = float(limit)
        self._spent = float(spent)
        self._lock = threading.Lock()
        # True when the library made this budget from a number; such a budget
        # belongs to one run of its owner, and the next run replaces it.
        self._per_run = False

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit - self._spent)

    @property
    def exhausted(self) -> bool:
        return self._spent >= self.limit

    def check(self, what: str = "") -> None:
        """Refuse to begin another call when there is nothing left.

        Parameters
        ----------
        what : str, optional
            What was about to begin, for the error's message.
        """
        if self.exhausted:
            raise BudgetExceeded(self._spent, self.limit, what)

    def charge(self, amount: float | None) -> None:
        """Record what a call cost. ``None`` is free — a provider that does
        not price its answers cannot be billed against a ceiling, and
        inventing a number would be worse than admitting it.

        Parameters
        ----------
        amount : float | None
            What the call cost; ``None`` when it could not be priced.
        """
        if amount is None:
            return
        import math
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) \
                or not math.isfinite(amount) or amount < 0:
            # A negative charge would give money back, and a NaN would make
            # every later comparison false.
            raise ValueError(f"a charge is a finite amount of 0 or more; got {amount!r}")
        if not amount:
            return
        with self._lock:
            self._spent += float(amount)

    def __repr__(self) -> str:
        return (f"Budget(limit={self.limit:.4f}, spent={self._spent:.4f}, "
                f"remaining={self.remaining:.4f})")


def as_budget(value) -> Budget | None:
    """Accept a `Budget`, a number, or None — so `max_cost=0.5` works and a
    shared object works, without two parameters that mean one thing."""
    if value is None or isinstance(value, Budget):
        return value
    budget = Budget(float(value))
    budget._per_run = True
    return budget


#: The budgets of the runs this call sits inside — a Chain's, a Pool's —
#: ambient for the duration of each, as the tenant is. A Skill checks and
#: charges its own and every one of these. Ambient rather than written onto
#: each step's `max_cost`, because a Pool's copies of one Chain share its steps
#: and writing and restoring them would race.
_AMBIENT: ContextVar = ContextVar("yait_aichain_budgets", default=())


@contextmanager
def lend(budget: Budget | None):
    """Make *budget* hold for every call inside the block, nested under any
    budget already in force."""
    if budget is None:
        yield
        return
    token = _AMBIENT.set(_AMBIENT.get() + (budget,))
    try:
        yield
    finally:
        _AMBIENT.reset(token)


def in_force(own: Budget | None) -> list:
    """*own* and every budget lent around the call, each once."""
    out: list = []
    for budget in (own, *_AMBIENT.get()):
        if budget is not None and all(budget is not b for b in out):
            out.append(budget)
    return out


def start_run(owner, spent: float = 0.0) -> None:
    """Give *owner* a fresh budget for this run, when its ceiling is a number.

    Only the budget *owner* made itself from a number is replaced. One it was
    handed — a shared `Budget`, or the one a Chain or Pool lends its steps for
    the duration of a run — is left alone, so a lent ceiling stays shared.
    *spent* seeds the new one: a resumed run has already spent that much.
    """
    current = getattr(owner, "max_cost", None)
    if current is not None and current._per_run and current is getattr(owner, "_own_budget", None):
        fresh = Budget(current.limit, spent=spent)
        fresh._per_run = True
        owner.max_cost = owner._own_budget = fresh
