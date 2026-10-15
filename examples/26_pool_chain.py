"""
26_pool_chain.py — The primitives nest: a Chain is a Pool's runner, one per URL, all at once.

Keys: ANTHROPIC_API_KEY; the pages are read with fetch("local"), which needs none.
"""

from yait_aichain import Chain, Model, Pool, Skill
from yait_aichain.chain import Step
from yait_aichain.tools import fetch

URLS = ["https://example.com", "https://www.iana.org/help/example-domains", "https://www.python.org"]

# Key: ANTHROPIC_API_KEY from the environment; or Model("claude-haiku-5-5", api_key="...")
summary = Skill(Model("claude-haiku-5-5"), name="summary",
                prompt="Summarise this page in one sentence:\n\n{text}")
reader = Chain([Step(fetch("local"), out="page", inputs={"url": "url"}),
                Step(summary, inputs={"text": "page.pages.0.text"})], name="reader")

results = Pool(reader, items=[{"url": u} for u in URLS], max_flows=3).run()
for url, answer, error in zip(URLS, results, results.errors):
    print(f"[{url}]\n{answer if error is None else 'failed: ' + error}\n")
