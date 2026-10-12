"""
A library exception arrives intact on the other side of a process boundary.

`ProcessPoolExecutor`, Celery and Ray pickle an exception to send it back.
Python rebuilds one by calling its class with `args`, and these classes take
their fields rather than their message — so the caller received a TypeError
from the unpickler (or a message formatted twice) instead of the error that
happened. Every public exception is round-tripped here, through `pickle` and
through `copy`, and must come back the same type, message and fields.
"""

import copy
import pickle
import unittest

from yait_aichain import (APIError, AuthenticationError, BudgetExceeded,
                          InsufficientCreditsError, InvalidRequestError,
                          InvalidStructuredOutputError, NetworkError, NotFoundError,
                          RateLimitError, ServerError, TaskFailedError,
                          TruncatedResponseError, UnsupportedOption)
from yait_aichain.state import Suspend


def _examples():
    yield BudgetExceeded(0.25, 0.10, "gpt-4o for skill x")
    for cls in (APIError, AuthenticationError, InsufficientCreditsError, InvalidRequestError,
                NetworkError, NotFoundError, RateLimitError, ServerError, TaskFailedError):
        yield cls(429, "slow down", retry_after=3.0)
    yield TruncatedResponseError("cut at 512 tokens", finish_reason="length", output_tokens=512)
    yield InvalidStructuredOutputError("not the schema", problems=["$.a missing"], value={"b": 1})
    yield UnsupportedOption("seed is not honoured", adaptations=["a"])
    yield Suspend("approve the refund", {"approved": "bool"}, hint={"on": "click"})


class TestEveryExceptionTravels(unittest.TestCase):

    def test_pickle_and_copy_keep_type_message_and_fields(self):
        for exc in _examples():
            for how, back in (("pickle", lambda e: pickle.loads(pickle.dumps(e))),
                              ("copy", copy.copy), ("deepcopy", copy.deepcopy)):
                with self.subTest(type(exc).__name__, how=how):
                    got = back(exc)
                    self.assertIs(type(got), type(exc))
                    self.assertEqual(str(got), str(exc))
                    self.assertEqual(vars(got), vars(exc))


if __name__ == "__main__":
    unittest.main()
