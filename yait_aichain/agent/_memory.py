"""
agent._memory
=============

``AgentMemory`` — a key-value store with a pluggable backend.

**The Agent does not use it.** An agent's state is its conversation; nothing
in the loop reads or writes this store, and ``AgentResult.memory`` is a field
that is never written. It is yours to use — from inside your own tools,
across runs, or as scratch space in a ``Chain``.

.. warning::
    Memory is **not** encrypted or redacted, and with ``FileBackend`` it is
    written to disk as plain JSON. Do not store secrets in it.

Pluggable backends
------------------
By default ``AgentMemory`` uses an in-process ``InMemoryBackend``. For state
that survives the process, supply a ``FileBackend``::

    from yait_aichain.agent import AgentMemory, FileBackend

    memory = AgentMemory(backend=FileBackend("~/.my_state.json"))
    memory.set("last_seen", "2026-10-10")
    memory.flush()                       # write it

:meth:`AgentMemory.reset` (not :meth:`~AgentMemory.clear`) restores the
loaded-plus-initial baseline without touching the backend.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Backend base class
# ---------------------------------------------------------------------------

class MemoryBackend:
    """
    Abstract base for memory persistence backends.

    Subclasses must implement :meth:`load`, :meth:`save`, and :meth:`clear`.
    """

    def load(self) -> dict:
        """Return the persisted key-value store as a plain dict."""
        raise NotImplementedError

    def save(self, data: dict) -> None:
        """Persist *data*, overwriting any previously saved state."""
        raise NotImplementedError

    def clear(self) -> None:
        """Erase all persisted data."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# In-memory backend (default)
# ---------------------------------------------------------------------------

class InMemoryBackend(MemoryBackend):
    """
    Volatile in-process backend.

    State lives only in ``self._data``; nothing is written to disk.

    This is the default backend: with no ``initial`` seed the baseline is
    empty, so each new ``AgentMemory`` starts with a clean store.
    """

    def __init__(self) -> None:
        self._data: dict = {}

    def load(self) -> dict:
        """Return a copy of the in-process store."""
        return dict(self._data)

    def save(self, data: dict) -> None:
        """Replace the in-process store with *data*."""
        self._data = dict(data)

    def clear(self) -> None:
        """Wipe the in-process store."""
        self._data = {}


# ---------------------------------------------------------------------------
# File-based backend
# ---------------------------------------------------------------------------

class FileBackend(MemoryBackend):
    """
    Durable JSON-file backend.

    State is written to a UTF-8 JSON file via an atomic rename so that a
    crash during ``save()`` never leaves a corrupt file.

    Parameters
    ----------
    path : str
        File path where the JSON state is stored.  ``~`` is expanded to the
        user's home directory.

    Examples
    --------
    ::

        backend = FileBackend("~/.agent_memory.json")
        memory  = AgentMemory(backend=backend)

        # Persist the current state:
        memory.flush()

        # After reset() the same instance starts from the flushed
        # checkpoint.  A new process restores the state the same way:
        memory2 = AgentMemory(backend=FileBackend("~/.agent_memory.json"))
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path).expanduser()

    def load(self) -> dict:
        """Load and return the JSON state, or ``{}`` on any error."""
        try:
            if self._path.exists():
                return json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
        return {}

    def save(self, data: dict) -> None:
        """
        Write *data* to the backing file atomically.

        Creates parent directories as needed.  Writes to a temporary file
        in the same directory, then renames it over the target path.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(data, indent=2, ensure_ascii=False)
        fd, tmp_path = tempfile.mkstemp(
            dir=self._path.parent, suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp_path, self._path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def clear(self) -> None:
        """Delete the backing file if it exists."""
        try:
            if self._path.exists():
                self._path.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# AgentMemory
# ---------------------------------------------------------------------------

class AgentMemory:
    """
    A key-value store with a pluggable backend.

    Stores any Python value: strings, dicts, lists, numbers.  Nothing reads
    it on its own — the agent loop does not use it — so what goes in comes
    out only where your own code asks for it.

    Parameters
    ----------
    initial : dict | None, optional
        Seed the memory with these key/value pairs.  Applied *on top of*
        any state loaded from the backend.
    backend : MemoryBackend | None, optional
        Persistence backend.  When omitted, an :class:`InMemoryBackend` is
        used (volatile, gone with the process).  Supply a
        :class:`FileBackend` to persist state across processes.

    Examples
    --------
    Ephemeral memory (default)::

        memory = AgentMemory({"language": "French", "topic": "fusion energy"})

    File-backed memory::

        from yait_aichain.agent import AgentMemory, FileBackend

        memory = AgentMemory(backend=FileBackend("~/.my_state.json"))
        memory.set("topic", "fusion energy")
        memory.flush()   # write it to the file
    """

    def __init__(
        self,
        initial: dict | None        = None,
        backend: MemoryBackend | None = None,
    ) -> None:
        self._backend: MemoryBackend  = backend or InMemoryBackend()
        loaded: dict[str, Any]        = dict(self._backend.load())
        if initial:
            loaded.update(initial)
        # Construction-time baseline: backend state + initial seed.
        # ``reset()`` restores this snapshot, so persistent state survives
        # while scratch values do not.
        self._baseline: dict[str, Any] = dict(loaded)
        self._store:    dict[str, Any] = dict(loaded)

    # ------------------------------------------------------------------
    # Read / Write
    # ------------------------------------------------------------------

    def set(self, key: str, value: Any) -> None:
        """Store *value* under *key*, overwriting any existing entry."""
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key*, or *default* if not present."""
        return self._store.get(key, default)

    def update(self, data: dict) -> None:
        """Merge *data* into the store (existing keys are overwritten)."""
        self._store.update(data)

    def delete(self, key: str) -> None:
        """Remove *key* from the store.  Silent no-op if absent."""
        self._store.pop(key, None)

    def reset(self) -> None:
        """
        Restore the in-process store to its construction-time baseline.

        The baseline is the state loaded from the backend plus any ``initial``
        seed.  Call it to discard scratch values while keeping persistent
        state (FileBackend contents, ``initial``).  The backend itself is
        never touched.
        """
        self._store = dict(self._baseline)

    def clear(self) -> None:
        """
        Remove all entries from the in-process store, the baseline, and the
        backend.  This is a destructive, explicit operation; :meth:`reset`
        is the one that keeps persistent state.
        """
        self._store.clear()
        self._baseline = {}
        self._backend.clear()

    def flush(self) -> None:
        """
        Persist the current in-process state to the backend without clearing.

        Call it to checkpoint the current state.  The flushed state also
        becomes the new baseline, so a later :meth:`reset` on the same
        instance returns to the checkpoint instead of the construction-time
        snapshot.
        """
        self._backend.save(self._store)
        self._baseline = dict(self._store)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def all(self) -> dict:
        """Return a shallow copy of the full store as a plain dict."""
        return dict(self._store)

    def keys(self):
        return self._store.keys()

    def values(self):
        return self._store.values()

    def items(self):
        return self._store.items()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        keys    = list(self._store.keys())
        backend = type(self._backend).__name__
        return f"AgentMemory(keys={keys}, backend={backend})"
