"""
utilities.vectors._tool — what the four vector tools share.

Each holds one store and refuses to be pointed at another collection: the
collection is the code's choice, never a model's.
"""

from __future__ import annotations

import copy

from ...tools._base import Tool
from ._base import VectorStore


class _VectorTool(Tool):

    def __init__(self, store: VectorStore) -> None:
        self._store = store
        # A filter is written in the backend's own dialect, and a model shown
        # only "a filter" writes one that the backend answers with an error.
        fields = self.parameters.get("properties", {}).get("options", {}).get("properties", {})
        dialect = getattr(getattr(store, "_backend", None), "FILTER", "")
        if "filter" in fields and isinstance(dialect, str) and dialect:
            self.parameters = copy.deepcopy(self.parameters)
            option = self.parameters["properties"]["options"]["properties"]["filter"]
            option["description"] += f" Written as: {dialect}"

    def _with_collection(self, opts: dict) -> VectorStore:
        """The store. Its collection is a setting, not a call's: a
        ``collection`` option is refused, so a model cannot read or write
        another collection than the one the code chose."""
        if "collection" in opts:
            raise ValueError(f"{self.name}: collection is the store's setting, not a call's — "
                             "VectorDB(…, collection=…) or Vectors(…, collection=…)")
        return self._store

    def __repr__(self) -> str:
        return f"{type(self).__name__}(store={self._store!r})"
