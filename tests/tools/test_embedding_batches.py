"""
Embedding batches are sized by what a provider refuses, and a write keeps
what it paid for.

Batches were cut by count alone: 2,048 chunks of 400 tokens is 819k tokens
against OpenAI's 300k, DashScope v3/v4 take ten rows where 25 were sent, and
50 records of 3,072 floats are 3.4 MB against Pinecone's 2 MB. Everything was
embedded before the first write, so a failure at batch k lost every
embedding already bought; and a provider answering with no vectors was a
success.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.tools.embed._base import Embedder
from yait_aichain.tools.embed._openai import EmbeddingOpenAI
from yait_aichain.tools.embed._qwen import EmbeddingQwen
from yait_aichain.utilities.vectors import PartialUpsert
from yait_aichain.utilities.vectors._base import VectorStore


class Stub(Embedder):
    provider = "stub"
    _ENV_KEY = "STUB_KEY"
    _MAX_BATCH = 100
    _MAX_TOKENS = 1000

    def __init__(self, dim=2, empty=False):
        super().__init__("m", api_key="k")
        self.sent, self.dim, self.empty = [], dim, empty

    def _embed_chunk(self, texts, *, input_type, dimensions, **opts):
        self.sent.append(len(texts))
        return ([] if self.empty else [[0.1] * self.dim for _ in texts]), None, {}


class Backend:
    def __init__(self, fail_on=None):
        self.writes, self.fail_on = [], fail_on

    def upsert(self, collection, recs, vectors):
        if self.fail_on is not None and len(self.writes) == self.fail_on:
            raise RuntimeError("HTTP 500")
        self.writes.append(len(recs))


class TestBatches(unittest.TestCase):

    def test_tokens_bound_a_batch_as_well_as_rows(self):
        e = Stub()
        e.embed(["x" * 900] * 10)                 # ~300 tokens each, 1000 a request
        self.assertTrue(all(n <= 3 for n in e.sent))
        self.assertEqual(sum(e.sent), 10)

    def test_openai_is_held_to_its_token_limit(self):
        e = EmbeddingOpenAI("text-embedding-3-small", api_key="k")
        sizes = [len(b) for b in e._batches(["word " * 400] * 2048)]
        self.assertGreater(len(sizes), 1)

    def test_dashscope_v4_takes_ten_rows(self):
        e = EmbeddingQwen("text-embedding-v4", api_key="k")
        self.assertEqual(max(len(b) for b in e._batches(["a"] * 25)), 10)

    def test_no_vectors_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "0 vectors"):
            Stub(empty=True).embed(["a"])


class TestUpsert(unittest.TestCase):

    def _records(self, n):
        return [{"id": str(i), "text": f"t{i}"} for i in range(n)]

    def test_a_failure_keeps_the_batches_written(self):
        store = VectorStore(Backend(fail_on=2), "c", embedder=Stub())
        with self.assertRaises(PartialUpsert) as ctx:
            store.upsert(self._records(10), batch_size=3)
        self.assertEqual(ctx.exception.written, 6)
        self.assertEqual(ctx.exception.total, 10)

    def test_embedding_happens_a_batch_at_a_time(self):
        e = Stub()
        VectorStore(Backend(), "c", embedder=e).upsert(self._records(10), batch_size=4)
        self.assertEqual(e.sent, [4, 4, 2])

    def test_wide_vectors_split_a_batch_by_bytes(self):
        backend = Backend()
        VectorStore(backend, "c", embedder=Stub(dim=3072)).upsert(self._records(50))
        self.assertGreater(len(backend.writes), 1)
        self.assertEqual(sum(backend.writes), 50)

    def test_batch_size_zero_is_refused(self):
        with self.assertRaises(ValueError):
            VectorStore(Backend(), "c", embedder=Stub()).upsert(self._records(1), batch_size=0)


if __name__ == "__main__":
    unittest.main()
