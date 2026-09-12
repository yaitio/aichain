"""
`score` means the same thing on every backend and every metric.

It did not. Chroma normalised its distances to [0, 1]; Qdrant and Pinecone
handed their raw numbers through. So one field carried a normalised score on
one backend, a cosine similarity in [-1, 1] on another, and — for the
euclidean metric — a **distance**, where lower is nearer. `query()` promised
"highest score = most similar" and that was false for two backends on one
metric.

Who this hurt: anyone comparing retrievers, and anyone who tuned a threshold
on one collection and moved it to another. Both are exactly the comparison
this library tells its users to make, and it is the same defect as the
provider defaults — three scales read as one, with nothing saying so.

`raw_score` keeps what the backend returned, because a conversion that throws
the original away cannot be checked.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yait_aichain.tools.vectordb._base import (METRICS,               # noqa: E402
                                               normalise_score)


class TestTheScale(unittest.TestCase):

    def test_cosine_lands_in_the_unit_interval(self):
        self.assertEqual(normalise_score(1.0, "cosine"), 1.0)
        self.assertEqual(normalise_score(0.0, "cosine"), 0.5)
        self.assertEqual(normalise_score(-1.0, "cosine"), 0.0)

    def test_euclidean_is_inverted_because_it_is_a_distance(self):
        """The direction is the whole reason this exists: 0 is a perfect
        match and larger is worse, so a raw distance sorted as a score ranks
        the *least* similar first."""
        near = normalise_score(0.0, "euclidean")
        far  = normalise_score(9.0, "euclidean")
        self.assertGreater(near, far)
        self.assertEqual(near, 1.0)

    def test_it_is_monotonic(self):
        """A conversion that reorders results would be worse than none."""
        for metric in ("cosine", "euclidean"):
            with self.subTest(metric=metric):
                raws = [0.0, 0.5, 1.0, 2.0]
                scores = [normalise_score(r, metric) for r in raws]
                if metric == "euclidean":
                    scores = list(reversed(scores))
                self.assertEqual(scores, sorted(scores))

    def test_dot_is_passed_through_and_says_so(self):
        """Unbounded and scale-dependent: it cannot be mapped onto [0, 1]
        without the vectors' magnitudes. A fabricated bound would look
        comparable when it is not."""
        self.assertEqual(normalise_score(37.5, "dot"), 37.5)

    def test_no_score_stays_no_score(self):
        """A fetch by id has none, and inventing one would make an exact
        lookup look like a ranked hit."""
        for metric in METRICS:
            self.assertIsNone(normalise_score(None, metric))


class TestEveryBackendAgrees(unittest.TestCase):
    """Built with `__new__` on purpose: this is about what `query()` does to
    a response, not about constructing a client."""

    def _qdrant(self, raw, metric):
        from yait_aichain.tools.vectordb.providers._qdrant import QdrantBackend
        b = QdrantBackend.__new__(QdrantBackend)
        b._metric = metric
        b._post = lambda *a, **k: {"result": [
            {"id": "1", "score": raw, "payload": {"text": "body"}}]}
        return b.query("col", [0.1])[0]

    def _pinecone(self, raw, metric):
        from yait_aichain.tools.vectordb.providers._pinecone import PineconeBackend
        b = PineconeBackend.__new__(PineconeBackend)
        b._metric = metric
        b._index_url = "https://h"
        b._headers = lambda: {}
        b._ctrl_http = MagicMock()
        b._ctrl_http.request.return_value = MagicMock(
            status=200,
            data=json.dumps({"matches": [
                {"id": "1", "score": raw, "metadata": {"text": "body"}}]}).encode())
        return b.query("col", [0.1])[0]

    def test_a_perfect_cosine_match_scores_one_everywhere(self):
        for name, build in (("qdrant", self._qdrant),
                            ("pinecone", self._pinecone)):
            with self.subTest(backend=name):
                self.assertEqual(build(1.0, "cosine").score, 1.0)

    def test_an_opposite_cosine_match_scores_zero_everywhere(self):
        for name, build in (("qdrant", self._qdrant),
                            ("pinecone", self._pinecone)):
            with self.subTest(backend=name):
                self.assertEqual(build(-1.0, "cosine").score, 0.0)

    def test_a_euclidean_distance_is_not_sorted_as_a_similarity(self):
        """The case that was broken. Raw, a distance of 9 outranks a distance
        of 0 in any descending sort."""
        for name, build in (("qdrant", self._qdrant),
                            ("pinecone", self._pinecone)):
            with self.subTest(backend=name):
                near = build(0.0, "euclidean").score
                far  = build(9.0, "euclidean").score
                self.assertGreater(near, far)

    def test_the_original_is_kept(self):
        for name, build in (("qdrant", self._qdrant),
                            ("pinecone", self._pinecone)):
            with self.subTest(backend=name):
                record = build(0.42, "cosine")
                self.assertEqual(record.raw_score, 0.42)
                self.assertNotEqual(record.score, record.raw_score)


class TestTheMetricIsDeclaredNotGuessed(unittest.TestCase):
    """Discovery was tried and withdrawn: asking the collection puts a
    network round trip with retries in front of the first query, which on an
    unreachable host is a minute of backoff before a search that would have
    failed fast. The suite went from 9 seconds to 68 and that is what showed
    it."""

    def test_both_backends_take_a_metric(self):
        import inspect
        from yait_aichain.tools.vectordb.providers._pinecone import PineconeBackend
        from yait_aichain.tools.vectordb.providers._qdrant import QdrantBackend
        for cls in (QdrantBackend, PineconeBackend):
            with self.subTest(backend=cls.__name__):
                self.assertIn("metric",
                              inspect.signature(cls.__init__).parameters)

    def test_and_default_to_the_same_one(self):
        from yait_aichain.tools.vectordb.providers._pinecone import PineconeBackend
        from yait_aichain.tools.vectordb.providers._qdrant import QdrantBackend
        self.assertEqual(QdrantBackend._metric, PineconeBackend._metric)


if __name__ == "__main__":
    unittest.main()
