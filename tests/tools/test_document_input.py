"""
A reranker and an embedder send text, and only text.

``rerank.run("hello world")`` sent eleven one-character documents; a
``VectorRecord`` went as its repr; a mapping with no ``text`` went whole,
metadata included; no query was sent as ``""``; ``top_n="5"`` went as a
string; an empty list still made a request; an embedder sent a ``chunk()``
dict whole, ``None`` as "None", and ``""``.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.tools.embed._base import Embedder
from yait_aichain.tools.rerank._base import RerankBase
from yait_aichain.utilities.vectors._base import VectorRecord


class Ranker(RerankBase):
    provider = "stub"
    _ENV_KEY = "STUB_KEY"

    def __init__(self):
        super().__init__("m", api_key="k")
        self.sent = []

    def _rerank_request(self, query, documents, top_n):
        self.sent.append((query, list(documents), top_n))
        return [{"index": i, "text": d, "score": 1.0 - i / 10} for i, d in enumerate(documents)], None, {}


class Vectors(Embedder):
    provider = "stub"
    _ENV_KEY = "STUB_KEY"

    def __init__(self):
        super().__init__("m", api_key="k")
        self.sent = []

    def _embed_chunk(self, texts, *, input_type, dimensions, **opts):
        self.sent.append(list(texts))
        return [[0.0, 1.0] for _ in texts], None, {}


class TestRerankInput(unittest.TestCase):

    def test_a_string_is_not_a_list_of_documents(self):
        with self.assertRaises(TypeError):
            Ranker().run("hello world", {"query": "q"})

    def test_a_record_sends_its_text_and_keeps_its_fields(self):
        r = Ranker()
        out = r.run([VectorRecord(id="1", text="alpha", metadata={"k": 1})], {"query": "q"})
        self.assertEqual(r.sent[0][1], ["alpha"])
        self.assertEqual(out[0]["id"], "1")

    def test_a_mapping_without_text_is_refused_not_sent(self):
        r = Ranker()
        with self.assertRaises(TypeError):
            r.run([{"email": "a@b.c"}], {"query": "q"})
        self.assertEqual(r.sent, [])

    def test_no_query_is_refused(self):
        with self.assertRaises(ValueError):
            Ranker().run(["a"])

    def test_top_n_is_a_whole_number(self):
        r = Ranker()
        r.run(["a", "b"], {"query": "q", "top_n": "1"})
        self.assertEqual(r.sent[0][2], 1)
        for bad in (0, True, "x", 1.5):
            with self.assertRaises(ValueError):
                r.run(["a"], {"query": "q", "top_n": bad})

    def test_no_documents_is_no_request(self):
        r = Ranker()
        self.assertEqual(r.run([], {"query": "q"}), [])
        self.assertEqual(r.sent, [])


class TestEmbedInput(unittest.TestCase):

    def test_a_chunk_dict_sends_its_text(self):
        e = Vectors()
        e.embed([{"text": "alpha", "index": 0}])
        self.assertEqual(e.sent, [["alpha"]])

    def test_none_and_empty_are_refused(self):
        with self.assertRaises(TypeError):
            Vectors().embed([None])
        with self.assertRaises(ValueError):
            Vectors().embed(["a", ""])


if __name__ == "__main__":
    unittest.main()
