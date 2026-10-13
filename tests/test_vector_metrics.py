"""
A store's metric is the one it was created with, and its scores are read on it.

``create(4, metric="euclidean")`` on Qdrant and Pinecone built a euclidean
collection and scored it as cosine, so an identical document scored 0.5 and a
far one 1.0. Chroma's ip was off by one, its default l2 space read as
cosine, and it gave no ``raw_score``; two backends' euclidean is the square.
An unknown metric became cosine.
"""

import unittest
from unittest.mock import MagicMock

from yait_aichain.clients._errors import APIError
from yait_aichain.utilities.vectors._base import VectorStore, normalise_score
from yait_aichain.utilities.vectors.providers._chroma import ChromaBackend
from yait_aichain.utilities.vectors.providers._pinecone import PineconeBackend
from yait_aichain.utilities.vectors.providers._qdrant import QdrantBackend


class _Embed:
    def embed(self, texts, input_type=None):
        return type("R", (), {"embeddings": [[1.0, 0.0] for _ in texts]})()


class TestCreateThenQuery(unittest.TestCase):

    def test_qdrant_scores_on_the_metric_it_created(self):
        b = QdrantBackend(url="http://localhost:6333")
        b._call = MagicMock(return_value={})
        store = VectorStore(b, "c", embedder=_Embed())
        store.create(2, metric="euclidean")
        b._call = MagicMock(return_value={"result": [{"id": 1, "score": 0.0, "payload": {"text": "same"}},
                                                     {"id": 2, "score": 9.0, "payload": {"text": "far"}}]})
        near, far = store.query("q")
        self.assertEqual(near.score, 1.0)
        self.assertLess(far.score, near.score)

    def test_pinecone_reads_an_existing_index_and_its_metric(self):
        b = PineconeBackend(api_key="k", url="https://idx.example.io", index="i")

        def call(method, url, body=None):
            if method == "POST" and url.endswith("/indexes"):
                raise APIError(409, "exists")
            return {"dimension": 2, "metric": "euclidean"}
        b._call = MagicMock(side_effect=call)
        VectorStore(b, "c").create(2, metric="cosine")
        self.assertEqual(b._metric, "euclidean")
        b._call = MagicMock(return_value={"matches": [{"id": "a", "score": 4.0, "metadata": {"text": "t"}}]})
        rec = b.query("c", [1.0, 0.0])[0]
        self.assertAlmostEqual(rec.score, 1 / 3)       # squared distance 4 is a distance of 2
        self.assertEqual(rec.raw_score, 4.0)

    def test_pinecone_refuses_an_index_of_another_dimension(self):
        b = PineconeBackend(api_key="k", url="https://idx.example.io", index="i")

        def call(method, url, body=None):
            if method == "POST":
                raise APIError(409, "exists")
            return {"dimension": 1536, "metric": "cosine"}
        b._call = MagicMock(side_effect=call)
        with self.assertRaisesRegex(ValueError, "1536"):
            b.create_collection("c", 3072)

    def test_an_unknown_metric_is_refused(self):
        with self.assertRaises(ValueError):
            VectorStore(QdrantBackend(url="http://localhost:6333"), "c").create(2, metric="hamming")
        with self.assertRaises(ValueError):
            PineconeBackend(api_key="k", metric="manhattan")


class TestChroma(unittest.TestCase):

    def _query(self, space, distance):
        b = ChromaBackend(url="http://localhost:8000")
        b._collection_cache["c"] = ("id1", space)
        b._call = MagicMock(return_value={"ids": [["a"]], "documents": [["t"]],
                                          "metadatas": [[{}]], "distances": [[distance]]})
        return b.query("c", [1.0])[0]

    def test_cosine_and_ip_and_l2_agree_with_the_other_backends(self):
        self.assertEqual(self._query("cosine", 0.0).score, normalise_score(1.0, "cosine"))
        self.assertEqual(self._query("ip", 0.0).score, 1.0)      # 1 − dot, so a dot of 1
        self.assertAlmostEqual(self._query("l2", 4.0).score, 1 / 3)
        self.assertEqual(self._query("l2", 4.0).raw_score, 4.0)

    def test_a_collection_without_a_space_is_l2(self):
        b = ChromaBackend(url="http://localhost:8000")
        b._get = MagicMock(return_value={"id": "x", "metadata": None})
        self.assertEqual(b._resolve("c"), ("x", "l2"))


class TestRecord(unittest.TestCase):

    def test_raw_score_is_serialised(self):
        from yait_aichain.utilities.vectors import VectorRecord
        self.assertEqual(VectorRecord(id="a", text="t", score=0.5, raw_score=0.0).to_dict()["raw_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
