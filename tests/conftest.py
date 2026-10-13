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


@pytest.fixture(autouse=True)
def _environment_is_restored():
    """Every test leaves `os.environ` as it found it.

    A test that set a key and popped it in `finally` removed the key other
    tests had set at import, and the suite passed only in the file order CI
    happens to use — mutation testing, which runs a subset in another order,
    was the first thing to notice. Restoring here makes order irrelevant for
    the environment, whatever an individual test does.
    """
    saved = dict(os.environ)
    yield
    if os.environ != saved:
        os.environ.clear()
        os.environ.update(saved)


@pytest.fixture(autouse=True)
def _offline_tests_stay_offline(request, monkeypatch):
    """An offline test cannot reach the network.

    Fifteen `list_models` tests mocked `urllib3.PoolManager`, and under an
    `HTTPS_PROXY` the client built a `ProxyManager` instead: the mock never
    applied, and the "offline" test asked the real API with the key "key".
    The proxy variables are cleared for an offline test, and a connection to
    anything but this machine is refused, so the next such test fails by
    name instead of by a provider's 401.
    """
    if request.node.get_closest_marker("live"):
        yield
        return
    for name in list(os.environ):
        if name.lower() in ("http_proxy", "https_proxy", "all_proxy"):
            monkeypatch.delenv(name, raising=False)
    import socket
    real = socket.socket.connect

    def connect(sock, address, *a, **k):
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, str) and host not in ("127.0.0.1", "::1", "localhost") \
                and sock.family in (socket.AF_INET, socket.AF_INET6):
            raise OSError(f"offline test tried to connect to {host}; mock the transport, "
                          "or mark the test's class Live")
        return real(sock, address, *a, **k)
    monkeypatch.setattr(socket.socket, "connect", connect)
    yield


def is_live(item) -> bool:
    cls = getattr(item, "cls", None)
    return cls is not None and cls.__name__.endswith("Live")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if is_live(item):
            item.add_marker(pytest.mark.live)
