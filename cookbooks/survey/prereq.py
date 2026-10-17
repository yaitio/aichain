"""Check keys, packages and services before anything is downloaded or paid for."""

import importlib.util
import os
import sys
import urllib.error
import urllib.request


def require(*, keys=(), packages=(), services=(), hints=None) -> None:
    """Print every gap with the command that fixes it, and exit 2 if there is one.

    keys: names, or tuples meaning "any one of"; hints: {name: what to do instead}.
    packages: (import name, pip line). services: (name, url, how to start it).
    """
    hints, gaps = hints or {}, []
    for key in keys:
        names = key if isinstance(key, tuple) else (key,)
        if not any(os.getenv(n) for n in names):
            label = " or ".join(names)
            gaps.append((label, hints.get(label, "set it in the environment")))
    for module, pip in packages:
        if importlib.util.find_spec(module) is None:
            gaps.append((module, pip))
    for name, url, start in services:
        try:
            urllib.request.urlopen(url, timeout=2).close()
        except urllib.error.HTTPError:
            pass                              # it answered; that is all this asks
        except OSError:
            gaps.append((f"{name} at {url}", start))
    for what, fix in gaps:
        print(f"missing: {what:<28} {fix}")
    if gaps:
        sys.exit(2)
