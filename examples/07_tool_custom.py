"""
07_tool_custom.py — Make any function a tool, and use it in a Chain.

Shows the pattern:
  1. @tool turns a function into a Tool — its name, docstring and annotated
     parameters are the schema
  2. Run it standalone
  3. Plug it into a Chain with a Skill

A Tool subclass is the other way, when a tool holds state or needs a schema
annotations cannot say — see docs/primitives/tools.md.

No API key needed for the tool (uses the free Frankfurter API).

Required env vars:
    ANTHROPIC_API_KEY
"""

import json
import os
import urllib.request

from yait_aichain.tools import tool
from yait_aichain.models import Model
from yait_aichain.skills import Skill
from yait_aichain.chain  import Chain

# ── Define a custom tool ──────────────────────────────────────────────────────

@tool(risk="read")
def currency_converter(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert an amount from one currency to another using live rates.

    amount: the amount to convert.
    from_currency: the source currency, e.g. USD.
    to_currency: the target currency, e.g. EUR.
    """
    # urllib.request follows the HTTPS_PROXY of the environment, where there is one.
    # The API's edge refuses urllib's default User-Agent (403), so the request names itself.
    url = f"https://api.frankfurter.dev/v1/latest?from={from_currency}&to={to_currency}"
    request = urllib.request.Request(url, headers={"User-Agent": "yait-aichain-example/1.0"})
    with urllib.request.urlopen(request, timeout=15) as response:
        rate = json.load(response)["rates"][to_currency]
    return {"amount": amount, "from": from_currency, "to": to_currency,
            "rate": rate, "converted": round(amount * rate, 2)}


# ── 1. Standalone usage ───────────────────────────────────────────────────────

result = currency_converter.run(amount=100, from_currency="USD", to_currency="EUR")
print("[standalone]")
print(result)

# ── 2. In a Chain with a Skill ────────────────────────────────────────────────

format_skill = Skill(
    model = Model("claude-haiku-4-5-20251001", api_key=os.getenv("ANTHROPIC_API_KEY")),
    input = {"messages": [{"role": "user", "parts": [
        "{amount} {from} = {converted} {to} (rate: {rate}). Format this as a friendly one-liner."
    ]}]},
)

# The tool returns a plain dict, so its fields become the chain's variables.
chain  = Chain(steps=[currency_converter, format_skill])
answer = chain.run(variables={"amount": 250, "from_currency": "GBP", "to_currency": "JPY"})
print("\n[chain]")
print(answer.output)
