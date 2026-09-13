"""
Pytest bootstrap: put the repository root on the path, and mark live tests.

Until 2026-09-12 this file revived the pre-`2.0` layout, aliasing every
sub-package into `sys.modules` under its old top-level name so that
`from models import Model` kept working. It was written as a temporary bridge
and lived for three months, and it was not free:

* `tests/agent/` is itself a package named `agent`, so inside that directory
  the alias lost to the real one and imports resolved to a different module
  depending on where the test file sat;
* a module imported under two names is two module objects with two copies of
  every module-level value, which is a class of Heisenbug nobody wants to
  debug in a test suite;
* and it hid the migration it was supposed to enable, because nothing ever
  failed.

All 37 files now import `yait_aichain.*`, which is what a reader of the tests
should see anyway: the suite ought to reach the library the way its users do.

**Live tests are a marker, not a substring.** CI selected them out with
`-k "not Live"`, and `-k` matches case-insensitively anywhere in a test's
name — so `test_nothing_claimed_goes_undelivered` and
`test_a_refusal_is_not_a_delivery` ("de*live*red") were deselected with the
live suite and never ran in CI. A test class whose name ends in `Live` is
marked `live` here; CI runs `-m "not live"`, the nightly job `-m live`.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def is_live(item) -> bool:
    cls = getattr(item, "cls", None)
    return cls is not None and cls.__name__.endswith("Live")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if is_live(item):
            item.add_marker(pytest.mark.live)
