"""
What each vector backend puts on the wire, and how a create meets a collection
that exists already.

The rules live in the shared base and a backend only translates; these pin
the translation, request by request, so moving a rule cannot move a byte.
Chroma's create took the space it asked for as the collection's even when
the collection existed with another, and never checked its dimension, while
Qdrant and Pinecone did both.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from yait_aichain.clients._errors import APIError
from yait_aichain.utilities.vectors._base import VectorRecord
from yait_aichain.utilities.vectors.providers._chroma import ChromaBackend
from yait_aichain.utilities.vectors.providers._pinecone import PineconeBackend
from yait_aichain.utilities.vectors.providers._qdrant import QdrantBackend

_RECS = [VectorRecord(id="doc_1", text="t", metadata={"k": "v"})]
_P = "/api/v2/tenants/default_tenant/databases/default_database"


def _calls(b):
    return [(c.args[0], c.args[1], c.args[2] if len(c.args) > 2 else None)
            for c in b._call.call_args_list]


class TestHeaders(unittest.TestCase):

    def test_each_backend_sends_its_key_its_own_way(self):
        base = {"Content-Type": "application/json", "Accept": "application/json"}
        self.assertEqual(QdrantBackend(url="http://q", api_key="k")._headers(), {**base, "api-key": "k"})
        self.assertEqual(QdrantBackend(url="http://q")._headers(), base)
        self.assertEqual(ChromaBackend(url="http://c", api_key="k")._headers(),
                         {**base, "Authorization": "Bearer k"})
        self.assertEqual(ChromaBackend(url="http://c")._headers(), base)
        self.assertEqual(PineconeBackend(api_key="k")._headers(), {**base, "Api-Key": "k"})

    def test_settings_come_from_the_environment(self):
        env = {"QDRANT_URL": "http://qq/", "QDRANT_API_KEY": "qk", "CHROMA_URL": "http://cc",
               "CHROMA_API_KEY": "ck", "CHROMA_TENANT": "t", "CHROMA_DATABASE": "d",
               "PINECONE_API_KEY": "pk", "PINECONE_INDEX": "pi"}
        with patch.dict(os.environ, env):
            q, c, p = QdrantBackend(), ChromaBackend(), PineconeBackend()
        self.assertEqual((q._url, q._api_key), ("http://qq", "qk"))
        self.assertEqual((c._url, c._api_key, c._prefix), ("http://cc", "ck", "/api/v2/tenants/t/databases/d"))
        self.assertEqual((p._api_key, p._index), ("pk", "pi"))
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(QdrantBackend()._url, "http://localhost:6333")
            self.assertEqual(ChromaBackend()._url, "http://localhost:8000")
            with self.assertRaisesRegex(ValueError, "PINECONE_API_KEY"):
                PineconeBackend()


class TestQdrantWire(unittest.TestCase):

    def setUp(self):
        self.b = QdrantBackend(url="http://q")
        self.b._call = MagicMock(return_value={})

    def test_requests(self):
        b = self.b
        b.create_collection("c", 3, "dot")
        b.query("c", [1.0], n=2, filter={"must": []})
        b.upsert("c", _RECS, [[1.0]])
        b.fetch("c", ["7"])
        b.delete("c", filter={"must": []})
        b.count("c")
        pid = b._call.call_args_list[2].args[2]["points"][0]["id"]
        self.assertEqual(_calls(b), [
            ("PUT", "/collections/c", {"vectors": {"size": 3, "distance": "Dot"}}),
            ("POST", "/collections/c/points/search",
             {"vector": [1.0], "limit": 2, "with_payload": True, "filter": {"must": []}}),
            ("PUT", "/collections/c/points?wait=true", {"points": [
                {"id": pid, "vector": [1.0], "payload": {"k": "v", "text": "t", "_aichain_id": "doc_1"}}]}),
            ("POST", "/collections/c/points", {"ids": [7], "with_payload": True}),
            ("POST", "/collections/c/points/delete?wait=true", {"filter": {"must": []}}),
            ("POST", "/collections/c/points/count", {"exact": True}),
        ])
        self.assertEqual(b._metric, "dot")

    def test_an_existing_collection_is_adopted_with_its_metric(self):
        def call(method, path, body=None):
            if method == "PUT":
                raise APIError(409, "exists")
            return {"result": {"config": {"params": {"vectors": {"size": 3, "distance": "Euclid"}}}}}
        self.b._call = MagicMock(side_effect=call)
        self.b.create_collection("c", 3, "cosine")
        self.assertEqual(self.b._metric, "euclidean")
        with self.assertRaisesRegex(ValueError, "dimension 3, not 4"):
            self.b.create_collection("c", 4, "cosine")


class TestPineconeWire(unittest.TestCase):

    def test_requests(self):
        b = PineconeBackend(api_key="k", url="https://h")
        b._call = MagicMock(return_value={"vectors": {"a": {"id": "a", "metadata": {"text": "x", "m": 1}}},
                                          "namespaces": {"c": {"vectorCount": 4}}})
        b.query("c", [1.0], n=2, filter={"k": {"$eq": 1}})
        b.upsert("c", _RECS, [[1.0]])
        got = b.fetch("c", ["a", "b"])
        b.delete("c", ids=["a"])
        self.assertEqual(b.count("c"), 4)
        self.assertEqual(_calls(b), [
            ("POST", "https://h/query", {"vector": [1.0], "topK": 2, "namespace": "c",
                                          "includeMetadata": True, "includeValues": False,
                                          "filter": {"k": {"$eq": 1}}}),
            ("POST", "https://h/vectors/upsert", {"vectors": [
                {"id": "doc_1", "values": [1.0], "metadata": {"k": "v", "text": "t"}}], "namespace": "c"}),
            ("GET", "https://h/vectors/fetch?ids=a&ids=b&namespace=c", None),
            ("POST", "https://h/vectors/delete", {"namespace": "c", "ids": ["a"]}),
            ("GET", "https://h/describe_index_stats", None),
        ])
        self.assertEqual([(r.id, r.text, r.metadata, r.score) for r in got], [("a", "x", {"m": 1}, None)])

    def test_the_host_is_discovered_once(self):
        b = PineconeBackend(api_key="k", index="i")
        b._call = MagicMock(side_effect=[{"host": "i-x.pinecone.io"}, {}, {}])
        b.count("c")
        b.count("c")
        self.assertEqual([c[1] for c in _calls(b)], ["https://api.pinecone.io/indexes/i",
                                                     "https://i-x.pinecone.io/describe_index_stats",
                                                     "https://i-x.pinecone.io/describe_index_stats"])

    def test_create_sends_the_index_spec(self):
        b = PineconeBackend(api_key="k", cloud="gcp", region="eu")
        b._call = MagicMock(return_value={"status": {"ready": True}})
        b.create_collection("ns", 8, "dot")
        self.assertEqual(_calls(b)[0], ("POST", "https://api.pinecone.io/indexes", {
            "name": "ns", "dimension": 8, "metric": "dotproduct",
            "spec": {"serverless": {"cloud": "gcp", "region": "eu"}}}))
        self.assertEqual(b._metric, "dot")


class TestChromaWire(unittest.TestCase):

    def setUp(self):
        self.b = ChromaBackend(url="http://c")

    def test_requests(self):
        b = self.b
        b._call = MagicMock(return_value={"id": "cid", "metadata": {"hnsw:space": "ip"}})
        b.create_collection("c", 3, "dot")
        b._call = MagicMock(return_value={"ids": [["a"]], "documents": [["x"]], "metadatas": [[{"m": 1}]],
                                          "distances": [[0.25]]})
        hit = b.query("c", [1.0], n=2, filter={"m": 1})[0]
        b._call = MagicMock(return_value={})
        b.upsert("c", _RECS, [[1.0]])
        b.fetch("c", ["a"])
        b.delete("c", ids=["a"])
        self.assertEqual(_calls(b), [
            ("POST", f"{_P}/collections/cid/upsert", {"ids": ["doc_1"], "embeddings": [[1.0]],
                                                     "documents": ["t"], "metadatas": [{"k": "v"}]}),
            ("POST", f"{_P}/collections/cid/get", {"ids": ["a"], "include": ["documents", "metadatas"]}),
            ("POST", f"{_P}/collections/cid/delete", {"ids": ["a"]}),
        ])
        self.assertEqual((hit.id, hit.text, hit.metadata, hit.raw_score, hit.score), ("a", "x", {"m": 1}, 0.25, 0.75))

    def test_create_asks_for_the_space(self):
        self.b._call = MagicMock(return_value={"id": "cid", "metadata": {"hnsw:space": "l2"}})
        self.b.create_collection("c", 3, "euclidean")
        self.assertEqual(_calls(self.b), [("POST", f"{_P}/collections", {
            "name": "c", "metadata": {"hnsw:space": "l2"}, "get_or_create": True})])
        self.assertEqual(self.b._collection_cache["c"], ("cid", "l2"))

    def test_an_existing_collection_keeps_its_own_space(self):
        """get_or_create answers with the collection that exists; its space,
        not the one asked for, is what its distances are on."""
        self.b._call = MagicMock(return_value={"id": "cid", "metadata": {"hnsw:space": "l2"}, "dimension": 3})
        self.b.create_collection("c", 3, "cosine")
        self.assertEqual(self.b._collection_cache["c"], ("cid", "l2"))

    def test_an_answer_without_a_space_keeps_the_one_asked_for(self):
        self.b._call = MagicMock(return_value={"id": "cid"})
        self.b.create_collection("c", 3, "cosine")
        self.assertEqual(self.b._collection_cache["c"], ("cid", "cosine"))

    def test_an_existing_collection_of_another_dimension_is_refused(self):
        self.b._call = MagicMock(return_value={"id": "cid", "metadata": {"hnsw:space": "cosine"},
                                               "dimension": 1536})
        with self.assertRaisesRegex(ValueError, "1536"):
            self.b.create_collection("c", 3072)
        self.assertNotIn("c", self.b._collection_cache)

    def test_a_metric_chroma_lacks_is_refused(self):
        self.b._call = MagicMock()
        with self.assertRaisesRegex(ValueError, "Chroma has no metric 'manhattan'"):
            self.b.create_collection("c", 3, "manhattan")
        self.b._call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
