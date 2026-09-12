"""
One Model, many tenants: the key belongs to the run, not to the object.

`PLAN.md` carried this as *"per-request context for multi-tenant secrets —
`RunContext` exists; whether it covers the injection story end to end is
unverified."* Measured 2026-09-12: it did not cover it at all. `RunContext`
reached `Chain` alone, and its own docstring calls its contents *non-secret*
— correctly, because `RunDocument.context` is serialised and `FileStore`
writes it to disk. A tenant's key placed there would be persisted in
plaintext.

So the secret is **resolved from** the context and never stored in it:
`Model(api_key=callable)` is asked once per request and handed whatever run is
in flight. The context stays what it says it is — a tenant's name and some
metadata — and the credential never touches the document.

The other half is reach. A `context=` parameter on `Skill`, on `Pool`, on
every `Tool` would be the environment leaking into the scenario, which
`VISION.md` names as the signal that a design has gone wrong. It is ambient
instead, in a `ContextVar`, which also has to be carried across `Pool`'s
threads by hand — a fact tested below, because it fails silently otherwise.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from yait_aichain import Chain, Model, Pool, Skill                  # noqa: E402
from yait_aichain.state import RunContext, current, using           # noqa: E402

_REPLY = json.dumps({
    "choices": [{"message": {"content": "ok"}}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}).encode()

KEYS = {"acme": "sk-acme", "globex": "sk-globex"}


def _by_tenant(context):
    return KEYS.get(getattr(context, "tenant", None), "")


def _model(**kw):
    m = Model("gpt-4o", **kw)
    m.client._post = MagicMock(return_value=_REPLY)
    return m


def _skill(model):
    return Skill(model=model,
                 input={"messages": [{"role": "user", "parts": ["hi"]}]})


class TestTheKeyFollowsTheRun(unittest.TestCase):

    def test_one_model_answers_to_two_tenants(self):
        """The whole point. Resolving at construction would bind the first
        tenant's credential to every later request."""
        model = _model(api_key=_by_tenant)
        seen = []
        for tenant in ("acme", "globex", "acme"):
            with using(RunContext(tenant=tenant)):
                seen.append(model.client.api_key)
        self.assertEqual(seen, ["sk-acme", "sk-globex", "sk-acme"])

    def test_it_reaches_the_wire(self):
        model = _model(api_key=_by_tenant)
        with using(RunContext(tenant="globex")):
            headers = model.client._auth_headers()
        self.assertIn("sk-globex", str(headers))

    def test_an_unresolved_key_fails_loudly(self):
        """There is no process-wide key to fall back on, and that is
        deliberate: a fallback would send one tenant's request under another's
        credential and the provider would answer normally."""
        model = _model(api_key=_by_tenant)
        with using(RunContext(tenant="nobody")):
            with self.assertRaises(ValueError) as caught:
                model.client._auth_headers()
        self.assertIn("nobody", str(caught.exception))

    def test_a_plain_key_still_works(self):
        """The lightness invariant: nothing moves for a caller who has one
        key and no tenants."""
        self.assertEqual(_model(api_key="sk-plain").client.api_key, "sk-plain")

    def test_the_environment_variable_still_works(self):
        os.environ["OPENAI_API_KEY"] = "sk-env"
        try:
            self.assertEqual(_model().client.api_key, "sk-env")
        finally:
            os.environ.pop("OPENAI_API_KEY", None)


class TestTheSecretIsNotInTheContext(unittest.TestCase):
    """The fact that settled the design. `RunDocument.context` is serialised
    and written to disk, so anything in `RunContext` is written with it."""

    def test_the_run_document_serialises_the_context(self):
        from yait_aichain.state import RunDocument
        doc = RunDocument.new("chain", ["a"], variables={})
        doc.context = RunContext(tenant="acme",
                                 metadata={"req": "r-1"}).to_dict()
        self.assertIn("context", doc.to_dict())
        self.assertIn("acme", json.dumps(doc.to_dict()))

    def test_the_resolver_is_never_stored_on_the_model(self):
        model = _model(api_key=_by_tenant)
        self.assertEqual(model._api_key, "")


class TestReachWithoutAParameter(unittest.TestCase):

    def test_a_chain_publishes_its_context_to_the_steps(self):
        """A `Skill` never took a `context=` and must not have to: the
        environment is injected from outside the scenario."""
        seen = []
        model = _model(api_key=lambda ctx: (seen.append(
            getattr(ctx, "tenant", None)) or "sk-acme"))
        Chain(steps=[(_skill(model), "out")]).run(
            context=RunContext(tenant="acme"))
        self.assertEqual(seen, ["acme"])

    def test_a_pool_carries_it_into_every_worker_thread(self):
        """`contextvars` do not cross a thread boundary on their own. Without
        the explicit copy every item resolves as if no run were in flight, and
        the failure reads as a missing key rather than a lost context."""
        seen = []
        model = _model(api_key=lambda ctx: (seen.append(
            getattr(ctx, "tenant", None)) or "sk-acme"))
        with using(RunContext(tenant="acme")):
            Pool(runner=_skill(model), items=[{}, {}, {}], max_flows=3).run()
        self.assertEqual(seen, ["acme"] * 3)

    def test_nesting_restores_the_outer_run(self):
        """Runs nest — an Agent inside a Chain step — so leaving the inner one
        must not clear the outer one's tenant for the rest of its life."""
        with using(RunContext(tenant="outer")):
            with using(RunContext(tenant="inner")):
                self.assertEqual(current().tenant, "inner")
            self.assertEqual(current().tenant, "outer")
        self.assertIsNone(current())


if __name__ == "__main__":
    unittest.main()
