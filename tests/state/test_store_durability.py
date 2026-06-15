"""
FileStore durability/robustness + ChromaBackend empty-delete guard (1.3.2).
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from state import FileStore


class TestFileStoreLoad(unittest.TestCase):

    def test_corrupt_file_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            store = FileStore(d)
            store.save("run-1", {"kind": "chain", "variables": {}})
            # truncate/corrupt the file behind the store's back
            with open(store._path("run-1"), "w") as fh:
                fh.write("{ not valid json")
            with self.assertRaises(ValueError):
                store.load("run-1")

    def test_missing_file_is_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(FileStore(d).load("nope"))

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            store = FileStore(d)
            doc = {"kind": "chain", "variables": {"a": 1}}
            store.save("run-2", doc)
            self.assertEqual(store.load("run-2"), doc)

    def test_non_json_value_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            store = FileStore(d)
            doc = {"kind": "chain", "variables": {"bad": {1, 2, 3}}}  # a set
            with self.assertRaises(ValueError) as ctx:
                store.save("run-3", doc)
            self.assertIn("JSON", str(ctx.exception))
            self.assertNotIsInstance(ctx.exception, TypeError)  # clear, not raw


class TestChromaEmptyDeleteGuard(unittest.TestCase):

    def test_empty_delete_refused(self):
        from tools.vectordb.providers._chroma import ChromaBackend
        backend = ChromaBackend(url="http://localhost:8000")
        with self.assertRaises(ValueError):
            backend.delete("docs")                 # no ids, no filter
        with self.assertRaises(ValueError):
            backend.delete("docs", ids=[], filter={})


if __name__ == "__main__":
    unittest.main()
