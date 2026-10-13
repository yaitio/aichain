"""
state._secret — a value a run uses and never writes down.
"""

from __future__ import annotations


class Secret:
    """
    A value a run uses and never writes down: a token a chain variable
    carries to a tool.

    ``REST`` recommends ``token_field`` from a chain variable, and every pause
    wrote that variable to the store, to ``SuspendedResult.document`` and to
    every history record; an Agent step was handed it in its GIVEN block, so it
    went to the model's provider. A ``Secret`` is left out of all four: its
    ``repr`` and ``str`` are ``***``, the chain keeps it in memory only, and
    a tool that takes one reads it with :meth:`reveal`. A run resumed in
    another process does not have it — pass it again in the signal.

    Parameters
    ----------
    value : str
        The secret.
    """

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        if isinstance(value, Secret):
            value = value.reveal()
        self._value = value

    def reveal(self) -> str:
        """The value itself, for the one call that needs it."""
        return self._value

    def __repr__(self) -> str:
        return "Secret('***')"

    def __str__(self) -> str:
        return "***"

    def __eq__(self, other) -> bool:
        return isinstance(other, Secret) and other._value == self._value

    def __hash__(self) -> int:
        return hash(("Secret", self._value))

    def __reduce__(self):
        # Not pickled with its value either: a pickle is a file too.
        raise TypeError("a Secret is not serialised; pass it again where it is needed")


def public(variables: dict) -> dict:
    """*variables* without their secrets — what may be written down or shown."""
    return {k: v for k, v in (variables or {}).items() if not isinstance(v, Secret)}


def reveal(value):
    """A Secret's value, anything else unchanged."""
    return value.reveal() if isinstance(value, Secret) else value
