"""
What the OpenAI-shaped embedders send, and how they read the answer.

OpenAI, Voyage and DashScope speak one wire and differ in data — the URL, the
name of the dimension field, whether floats are asked for, whether
``input_type`` goes. Pinned per provider, so the shared path cannot drift one
of them.
"""

import os
import unittest
from unittest.mock import patch

from yait_aichain.tools import embed

KEYS = {"OPENAI_API_KEY": "k", "VOYAGE_API_KEY": "k", "DASHSCOPE_API_KEY": "k"}
ANSWER = {"data": [{"embedding": [0.2], "index": 1}, {"embedding": [0.1], "index": 0}],
          "usage": {"total_tokens": 7}}


def _sent(service, model, input_type=None, **settings):
    if service == "qwen":                  # region= is EmbeddingQwen's, not the factory's
        from yait_aichain.tools.embed._qwen import EmbeddingQwen
        e = EmbeddingQwen(model, **settings)
    else:
        e = embed(service, model=model, **settings)
    with patch.object(type(e), "_request", return_value=ANSWER) as req:
        result = e.run(["a", "b"], {"input_type": input_type} if input_type else None)
    url, body, headers = req.call_args.args
    return url, body, headers, result


@patch.dict(os.environ, KEYS)
class TestTheWire(unittest.TestCase):

    def test_openai(self):
        url, body, headers, _ = _sent("openai", "text-embedding-3-small", "query", dimensions=256)
        self.assertEqual(url, "https://api.openai.com/v1/embeddings")
        self.assertEqual(body, {"model": "text-embedding-3-small", "input": ["a", "b"],
                                "encoding_format": "float", "dimensions": 256})
        self.assertEqual(headers, {"Authorization": "Bearer k"})

    def test_voyage(self):
        url, body, _, _ = _sent("voyage", "voyage-3", "document", dimensions=512)
        self.assertEqual(url, "https://api.voyageai.com/v1/embeddings")
        self.assertEqual(body, {"model": "voyage-3", "input": ["a", "b"],
                                "output_dimension": 512, "input_type": "document"})

    def test_dashscope(self):
        url, body, _, _ = _sent("qwen", "text-embedding-v4", "query", region="ap")
        self.assertTrue(url.endswith("/compatible-mode/v1/embeddings"))
        self.assertEqual(body, {"model": "text-embedding-v4", "input": ["a", "b"],
                                "encoding_format": "float", "input_type": "query"})

    def test_nothing_unasked_is_sent(self):
        _, body, _, _ = _sent("voyage", "voyage-3")
        self.assertEqual(body, {"model": "voyage-3", "input": ["a", "b"]})

    def test_the_answer_is_read_by_index(self):
        _, _, _, result = _sent("openai", "text-embedding-3-small")
        self.assertEqual((result.embeddings, result.total_tokens), ([[0.1], [0.2]], 7))


if __name__ == "__main__":
    unittest.main()
