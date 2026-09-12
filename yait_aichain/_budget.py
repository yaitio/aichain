"""
_budget — a ceiling in money, shared by everything under one run.

``max_tokens`` bounds one reply. Nobody budgets in replies: a chain of ten
steps, a pool over a thousand items and an agent that decides how many turns
it needs are all unbounded in the only unit that appears on the invoice. The
agent has had ``cost_budget`` since 2.0 and the other three primitives had
nothing at all, so a runaway `Pool` was visible on the bill and nowhere else.

**One object, passed down, decremented by whoever spends.** That is the whole
mechanism, and it is why a budget handed to a `Chain` covers the agent inside
it: they share the object rather than each holding a copy of a number. A copy
would let every level spend the full amount, which is the failure mode this
shape exists to avoid.

Where the money is actually counted is ``Skill``, because ``Skill`` is the one
primitive that talks to a model — `Agent` builds Skills, `Chain` and `Pool`
run them. A ceiling anywhere else would be a second place to keep in sync.

**What it bounds, exactly.** The length of a reply is not known before it is
paid for, so this bounds *do not begin another call* and not *never exceed by
a cent*. The same caveat `cost_budget` carries, for the same reason, and it is
worth saying twice: a budget that promised the second thing would be lying
about the one number a caller checks.
"""

from __future__ import annotations

import threading


class BudgetExceeded(RuntimeError):
    """A run stopped because its money ceiling was reached.

    A raise rather than a partial result. Returning what was finished with a
    flag set is how a truncated run gets read as a complete one — the same
    silent-success shape this library has spent its 2.x line removing — and a
    caller who wants the partial output can catch this and read
    ``last_usage``.
    """

    def __init__(self, spent: float, limit: float, what: str = "") -> None:
        super().__init__(
            f"budget exhausted{' on ' + what if what else ''}: "
            f"${spent:.4f} spent of ${limit:.4f}. Raise max_cost, or catch "
            "BudgetExceeded and read what completed.")
        self.spent = spent
        self.limit = limit


class Budget:
    """A ceiling in dollars, and what has been spent against it.

    Thread-safe because `Pool` runs its items concurrently and a budget that
    can be overspent by a race is not a budget.

    ::

        budget = Budget(0.50)
        Chain(steps=[...], budget=budget).run()
        budget.spent        # what it cost
        budget.remaining    # what is left
    """

    def __init__(self, limit: float) -> None:
        if limit <= 0:
            raise ValueError(f"a budget must be positive; got {limit!r}")
        self.limit = float(limit)
        self._spent = 0.0
        self._lock = threading.Lock()

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
        """Refuse to begin another call when there is nothing left."""
        if self.exhausted:
            raise BudgetExceeded(self._spent, self.limit, what)

    def charge(self, amount: "float | None") -> None:
        """Record what a call cost. ``None`` is free — a provider that does
        not price its answers cannot be billed against a ceiling, and
        inventing a number would be worse than admitting it."""
        if not amount:
            return
        with self._lock:
            self._spent += float(amount)

    def __repr__(self) -> str:
        return (f"Budget(limit={self.limit:.4f}, spent={self._spent:.4f}, "
                f"remaining={self.remaining:.4f})")


def as_budget(value) -> "Budget | None":
    """Accept a `Budget`, a number, or None — so `max_cost=0.5` works and a
    shared object works, without two parameters that mean one thing."""
    if value is None or isinstance(value, Budget):
        return value
    return Budget(float(value))
