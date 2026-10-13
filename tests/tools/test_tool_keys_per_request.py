"""
A tool's key may be resolved per request, as a model's is.

``Model(api_key=callable)`` asks the resolver with the run's context on each
request; a tool froze what it was given and sent ``Bearer <function …>``.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.state import RunContext, using
from yait_aichain.tools import embed, rerank


def _keys(ctx):
    return {"acme": "key-acme", "globex": "key-globex"}.get(getattr(ctx, "tenant", None), "")


def _reply(body):
    r = MagicMock(status=200, headers={})
    r.data = json.dumps(body).encode()
    r.read = MagicMock(return_value=r.data)
    return r


class TestKeysPerRequest(unittest.TestCase):

    def _auth(self, tool):
        return tool._http.request.call_args.kwargs["headers"]["Authorization"]

    def test_an_embedder_sends_each_tenants_key(self):
        e = embed("openai", model="text-embedding-3-small", api_key=_keys)
        e._http = MagicMock()
        e._http.request.return_value = _reply({"data": [{"embedding": [0.1], "index": 0}],
                                               "usage": {"total_tokens": 1}})
        for tenant in ("acme", "globex"):
            with using(RunContext(tenant=tenant)):
                e.embed("hi")
            self.assertEqual(self._auth(e), f"Bearer key-{tenant}")

    def test_a_reranker_sends_each_tenants_key(self):
        r = rerank("cohere", model="rerank-v3.5", api_key=_keys)
        r._http = MagicMock()
        r._http.request.return_value = _reply({"results": [{"index": 0, "relevance_score": 0.5}]})
        with using(RunContext(tenant="globex")):
            r.run(["a"], {"query": "q"})
        self.assertEqual(self._auth(r), "Bearer key-globex")

    def test_no_key_for_the_tenant_is_an_error_not_a_fallback(self):
        e = embed("openai", model="text-embedding-3-small", api_key=_keys)
        e._http = MagicMock()
        with using(RunContext(tenant="initech")), self.assertRaisesRegex(ValueError, "initech"):
            e.embed("hi")
        e._http.request.assert_not_called()

    def test_a_key_that_is_neither_is_refused(self):
        with self.assertRaises(TypeError):
            embed("openai", model="text-embedding-3-small", api_key=12345)


if __name__ == "__main__":
    unittest.main()
