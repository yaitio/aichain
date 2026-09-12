"""
Pytest bootstrap: put the repository root on the path, and nothing else.

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
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
