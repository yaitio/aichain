"""
37_tenant_keys.py — One Model serves every tenant: each request carries its customer's key.

Keys: none. Runs offline under recorded(); the keys below are placeholders.
"""

import json
import os
import tempfile
from pathlib import Path

from yait_aichain import Chain, Model, Skill
from yait_aichain.state import FileStore, RunContext
from yait_aichain.testing import recorded
from yait_aichain.tools import wait

OUT = Path(os.getenv("AICHAIN_EXAMPLES_DIR", tempfile.gettempdir())) / "aichain-examples"
KEYS = {"tenant-a": "sk-tenant-a-0000000000001111", "tenant-b": "sk-tenant-b-0000000000002222"}

# The key is looked up per request from the run's context, which names the
# tenant and never holds the key — the context is saved with a parked run.
model = Model("gpt-4o", api_key=lambda context: KEYS[context.tenant])
store = FileStore(str(OUT / "tenant-runs"))
chain = Chain([Skill(model, prompt="Summarise: {text}", name="summary"),
               wait(reason="A person reviews the summary.")], store=store)

with recorded(model, reply="A short summary.") as rec:
    parked = [chain.run(variables={"text": "Quarterly revenue rose 4%."},
                        context=RunContext(tenant=tenant)) for tenant in KEYS]

for tenant, request in zip(KEYS, rec.requests):
    print(f"{tenant} sent {request.headers['Authorization']}")
saved = [json.dumps(store.load(run.run_id)) for run in parked]
print(f"tenants in store: {[json.loads(doc)['context']['tenant'] for doc in saved]}")
print(f"keys in store: {sum(doc.count('sk-') for doc in saved)}")
