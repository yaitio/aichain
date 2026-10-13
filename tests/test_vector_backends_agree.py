"""
Three vector backends, one meaning for each operation.

``delete(ids, filter)`` meant three things — Qdrant dropped the filter,
Chroma took both, Pinecone with neither deleted the namespace. Qdrant wrote
without ``wait=true``, so a query straight after found nothing. A typo in a
backend's settings was swallowed by ``**kwargs``. ``"007"`` and ``"7"`` were
one Qdrant point and ``"²"`` raised; a record with no id or with metadata of
None failed deep inside a write.
"""

import unittest
from unittest.mock import MagicMock

from yait_aichain.utilities.vectors._base import VectorStore
from yait_aichain.utilities.vectors.providers._chroma import ChromaBackend
from yait_aichain.utilities.vectors.providers._pinecone import PineconeBackend
from yait_aichain.utilities.vectors.providers._qdrant import QdrantBackend, _point_id


def _backends():
    q = QdrantBackend(url="http://localhost:6333")
    p = PineconeBackend(api_key="k", url="https://idx.example.io")
    c = ChromaBackend(url="http://localhost:8000")
    c._collection_cache["c"] = ("id1", "cosine")
    for b in (q, p, c):
        b._call = MagicMock(return_value={})
    return q, p, c


class TestDelete(unittest.TestCase):

    def test_ids_and_filter_together_are_refused_everywhere(self):
        for b in _backends():
            with self.subTest(b=type(b).__name__), self.assertRaises(ValueError):
                b.delete("c", ids=["a"], filter={"k": "v"})
            b._call.assert_not_called()

    def test_nothing_named_is_refused_everywhere(self):
        for b in _backends():
            with self.subTest(b=type(b).__name__), self.assertRaises(ValueError):
                b.delete("c")
            b._call.assert_not_called()

    def test_the_store_refuses_both_as_well(self):
        q, _, _ = _backends()
        with self.assertRaises(ValueError):
            VectorStore(q, "c").delete(ids=["a"], filter={"k": "v"})


class TestQdrant(unittest.TestCase):

    def test_writes_wait_until_searchable(self):
        q, _, _ = _backends()
        q.upsert("c", [], [])
        q.delete("c", ids=["1"])
        paths = [call.args[1] for call in q._call.call_args_list]
        self.assertTrue(all(p.endswith("?wait=true") for p in paths), paths)

    def test_only_an_integers_own_spelling_is_an_integer_id(self):
        self.assertEqual(_point_id("7"), 7)
        self.assertNotEqual(_point_id("007"), 7)
        self.assertIsInstance(_point_id("²"), str)


class TestSettings(unittest.TestCase):

    def test_a_typo_in_a_setting_is_an_error(self):
        with self.assertRaises(TypeError):
            PineconeBackend(api_key="k", indx="i")
        with self.assertRaises(TypeError):
            QdrantBackend(url="http://localhost:6333", regoin="eu")


class TestRecords(unittest.TestCase):

    def test_a_bad_record_is_refused_before_a_write(self):
        q, _, _ = _backends()
        store = VectorStore(q, "c")
        for bad in ({"text": "t", "vector": [1.0]},
                    {"id": "a", "vector": [1.0]},
                    {"id": "a", "text": "t", "metadata": ["x"], "vector": [1.0]}):
            with self.subTest(bad=bad), self.assertRaises((ValueError, TypeError)):
                store.upsert([bad])
        q._call.assert_not_called()

    def test_metadata_of_none_is_empty(self):
        q, _, _ = _backends()
        self.assertEqual(VectorStore(q, "c").upsert([{"id": "a", "text": "t", "metadata": None,
                                                      "vector": [1.0]}]), 1)


class TestPineconeReady(unittest.TestCase):

    def test_create_waits_for_the_index(self):
        _, p, _ = _backends()
        p._index = "i"
        answers = iter([{}, {"status": {"ready": False}}, {"status": {"ready": True}}])
        p._call = MagicMock(side_effect=lambda *a, **k: next(answers))
        import time
        real, time.sleep = time.sleep, lambda s: None
        try:
            p.create_collection("c", 2)
        finally:
            time.sleep = real
        self.assertEqual(p._call.call_count, 3)


if __name__ == "__main__":
    unittest.main()
