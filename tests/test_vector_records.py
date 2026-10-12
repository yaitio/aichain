"""
A record reads back the way it was written, on every backend.

The payload parsing was written once per call site and had drifted: a query
read a ``document`` key as the text when ``text`` was absent and a fetch did
not, a metadata key named ``text`` overwrote the text, a Chroma document of
None broke a record, and a Chroma collection recreated under the same name
answered 404 for the life of the process. One reader per shape now, and the
keys a store keeps beside the metadata are refused in it.
"""

import unittest
from unittest.mock import MagicMock

from yait_aichain.clients._errors import NotFoundError
from yait_aichain.utilities.vectors import VectorDeleteTool, VectorQueryTool
from yait_aichain.utilities.vectors._base import VectorBackend, VectorStore
from yait_aichain.utilities.vectors.providers._chroma import ChromaBackend
from yait_aichain.utilities.vectors.providers._pinecone import PineconeBackend
from yait_aichain.utilities.vectors.providers._qdrant import QdrantBackend


def _qdrant(answer):
    b = QdrantBackend(url="http://localhost:6333")
    b._call = MagicMock(return_value=answer)
    return b


def _pinecone(answer):
    b = PineconeBackend(api_key="k", url="https://idx.example.io")
    b._call = MagicMock(return_value=answer)
    return b


class TestOneReader(unittest.TestCase):

    def test_query_and_fetch_read_the_same_payload_the_same_way(self):
        legacy = {"document": "old text", "source": "a"}
        q = _qdrant({"result": [{"id": 1, "score": 0.5, "payload": legacy}]})
        self.assertEqual([(r.text, r.metadata) for r in q.query("c", [0.1])],
                         [("old text", {"source": "a"})])
        self.assertEqual([(r.text, r.metadata) for r in q.fetch("c", ["1"])],
                         [("old text", {"source": "a"})])
        p = _pinecone({"vectors": {"x": {"metadata": legacy}}})
        self.assertEqual([r.text for r in p.fetch("c", ["x"])], ["old text"])

    def test_qdrant_restores_the_string_id(self):
        q = _qdrant({"result": [{"id": "uuid", "payload": {"text": "t", "_aichain_id": "doc_1"}}]})
        self.assertEqual([(r.id, r.metadata) for r in q.fetch("c", ["doc_1"])], [("doc_1", {})])

    def test_a_chroma_document_of_none_is_empty_text(self):
        b = ChromaBackend(url="http://localhost:8000")
        b._collection_cache["c"] = ("id1", "cosine")
        b._call = MagicMock(return_value={"ids": ["a", "b"], "documents": [None, "x"], "metadatas": [None]})
        self.assertEqual([(r.id, r.text, r.metadata) for r in b.fetch("c", ["a", "b"])],
                         [("a", "", {}), ("b", "x", {})])


class TestReservedKeys(unittest.TestCase):

    def test_metadata_may_not_use_the_stores_own_keys(self):
        store = VectorStore(_qdrant({}), "c")
        for key in ("text", "_aichain_id"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, key):
                    store.upsert([{"id": "1", "text": "t", "vector": [0.1], "metadata": {key: "x"}}])


class TestChromaCache(unittest.TestCase):

    def test_a_recreated_collection_is_looked_up_again(self):
        b = ChromaBackend(url="http://localhost:8000")
        b._collection_cache["c"] = ("stale", "cosine")
        b._call = MagicMock(side_effect=[NotFoundError(404, "gone"),
                                         {"id": "fresh", "metadata": {}},
                                         7])
        self.assertEqual(b.count("c"), 7)
        self.assertEqual(b._collection_cache["c"][0], "fresh")
        self.assertIn("/collections/fresh/count", b._call.call_args.args[1])


class TestWhatAModelIsShown(unittest.TestCase):

    def test_the_filter_dialect_is_the_backends(self):
        for backend, word in ((_qdrant({}), "must"), (_pinecone({}), "$eq"),
                              (ChromaBackend(url="http://localhost:8000"), "where")):
            store = VectorStore(backend, "c")
            for tool in (VectorQueryTool(store), VectorDeleteTool(store)):
                with self.subTest(backend=backend.provider, tool=tool.name):
                    said = tool.parameters["properties"]["options"]["properties"]["filter"]["description"]
                    self.assertIn(word, said)

    def test_n_is_bounded(self):
        tool = VectorQueryTool(VectorStore(_qdrant({}), "c"))
        n = tool.parameters["properties"]["options"]["properties"]["n"]
        self.assertEqual((n["minimum"], n["maximum"]), (1, 100))
        for bad in (0, 10_000):
            with self.assertRaisesRegex(ValueError, "between 1 and 100"):
                tool.run("q", {"n": bad})


class TestTheBackendIsAbstract(unittest.TestCase):

    def test_a_backend_missing_a_method_cannot_be_made(self):
        class Half(VectorBackend):
            provider = "half"

            def query(self, collection, vector, n=5, filter=None):
                return []

        with self.assertRaises(TypeError):
            Half(url="http://x")


if __name__ == "__main__":
    unittest.main()
