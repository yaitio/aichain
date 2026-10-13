"""
state._store — StateStore + built-in backends
==============================================

Persistence for suspended runs: a ``run_id`` → run-document (a JSON-serialisable
dict) mapping. The interface is deliberately tiny (``save``/``load``/``delete``)
so any KV / document store fits: in-memory (default), a local directory
(``FileStore``), or — by subclassing — S3, DynamoDB, Redis, Mongo, …

The store holds only *suspended* runs; a completed run is deleted. Full audit
of every run is the observability concern (M3), routed elsewhere.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
import uuid


class StateStore:
    """Abstract persistence for run documents, keyed by ``run_id``."""

    def save(self, run_id: str, document: dict) -> None:
        raise NotImplementedError

    def load(self, run_id: str) -> dict | None:
        """Return the stored document, or ``None`` if there is none."""
        raise NotImplementedError

    def delete(self, run_id: str) -> None:
        raise NotImplementedError

    def take(self, run_id: str) -> dict | None:
        """Remove the document and return it, as one step: the claim a resume
        makes before it acts, so that two deliveries of one signal — an SQS
        message seen twice, a webhook retried — cannot both run the approved
        action. ``None`` when there is nothing to take, which a second
        delivery reads as "already handled".

        This default loads then deletes, which is **not** atomic. A store
        shared between processes overrides it with the backend's own claim:
        a conditional delete in DynamoDB, ``GETDEL`` in Redis.

        Parameters
        ----------
        run_id : str
            The run to claim.
        """
        document = self.load(run_id)
        if document is not None:
            self.delete(run_id)
        return document


def _as_json(document: dict) -> str:
    try:
        return json.dumps(document, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        # A chain variable or step output that reached the suspend point is not
        # JSON-serialisable; a store cannot park it.
        raise ValueError(
            "Cannot persist the suspended run: a variable or step output is not "
            "JSON-serialisable (e.g. bytes, a set, or a custom object). Stores "
            f"require JSON-safe values at suspend points. ({exc})"
        ) from exc


class InMemoryStore(StateStore):
    """
    Process-local store (the default). Survives suspend→resume **within one
    process** only; for cross-process / serverless use a shared store
    (``FileStore`` or a custom S3/Dynamo backend).
    """

    def __init__(self) -> None:
        self._runs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def save(self, run_id: str, document: dict) -> None:
        # Through JSON, as every persistent store does: a set, bytes or an int
        # key worked here and broke in production on the first FileStore.
        self._runs[run_id] = json.loads(_as_json(document))

    def take(self, run_id: str) -> dict | None:
        """Claimed under a lock: of two threads taking one run, one gets it.

        Parameters
        ----------
        run_id : str
            The run to claim.
        """
        with self._lock:
            return self._runs.pop(run_id, None)

    def load(self, run_id: str) -> dict | None:
        doc = self._runs.get(run_id)
        return copy.deepcopy(doc) if doc is not None else None

    def delete(self, run_id: str) -> None:
        self._runs.pop(run_id, None)


class FileStore(StateStore):
    """
    Store each run as ``<dir>/<run_id>.json`` (atomic write). Survives a
    process restart, so it is the simplest persistent backend for serverless
    when the directory is on shared storage (EFS, a mounted volume, …).

    Parameters
    ----------
    directory : str
        Where the files go; created if it does not exist.
    """

    def __init__(self, directory: str) -> None:
        # `~/runs` was a folder named "~" in the working directory.
        self._dir = os.path.expanduser(directory)
        os.makedirs(self._dir, exist_ok=True)

    def _path(self, run_id: str) -> str:
        # Keep the file name safe regardless of the run_id source.
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in run_id)
        return os.path.join(self._dir, f"{safe}.json")

    def save(self, run_id: str, document: dict) -> None:
        path = self._path(run_id)
        # Serialised before the temporary file exists: a document that could
        # not be written left an open descriptor and a stray file behind.
        payload = _as_json(document)
        fd, tmp = tempfile.mkstemp(dir=self._dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())     # durable on disk before the rename
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def load(self, run_id: str) -> dict | None:
        try:
            with open(self._path(run_id), encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            return None
        except json.JSONDecodeError as exc:
            # A present-but-corrupt file (e.g. a crash mid-write before fsync
            # landed) is not "no such run" — surface it clearly rather than
            # leaking a raw JSONDecodeError or silently losing the run.
            raise ValueError(
                f"Corrupt run document for {run_id!r} at "
                f"{self._path(run_id)!r}: {exc}"
            ) from exc

    def delete(self, run_id: str) -> None:
        try:
            os.unlink(self._path(run_id))
        except FileNotFoundError:
            pass

    def take(self, run_id: str) -> dict | None:
        """Claimed by a rename, which the filesystem makes atomic: of two
        processes taking one run, one rename succeeds and the other finds no
        file.

        Parameters
        ----------
        run_id : str
            The run to claim.
        """
        claimed = f"{self._path(run_id)}.{uuid.uuid4().hex}.taken"
        try:
            os.rename(self._path(run_id), claimed)
        except FileNotFoundError:
            return None
        try:
            with open(claimed, encoding="utf-8") as fh:
                return json.load(fh)
        finally:
            os.unlink(claimed)
