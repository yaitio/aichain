"""
07_tool_custom.py — Define your own tool and use it in a Chain.

Shows the full pattern:
  1. Define a Tool subclass
  2. Run it standalone
  3. Plug it into a Chain with a Skill

No API key needed for the tool (uses free frankfurter.app).

Required env vars:
    ANTHROPIC_API_KEY
"""

import os, sys, json

import urllib3
from yait_aichain.tools import Tool
from yait_aichain.models import Model
from yait_aichain.skills import Skill
from yait_aichain.chain  import Chain

# ── Define a custom tool ──────────────────────────────────────────────────────

class CurrencyConverterTool(Tool):
    name        = "currency_converter"
    description = "Convert an amount from one currency to another using live rates."
    parameters  = {
        "type": "object",
        "properties": {
            "amount":        {"type": "number", "description": "Amount to convert."},
            "from_currency": {"type": "string", "description": "Source currency, e.g. USD."},
            "to_currency":   {"type": "string", "description": "Target currency, e.g. EUR."},
        },
        "required": ["amount", "from_currency", "to_currency"],
    }

    def run(self, amount: float, from_currency: str, to_currency: str, options=None) -> dict:
        http     = urllib3.PoolManager()
        response = http.request(
            "GET",
            f"https://api.frankfurter.app/latest?from={from_currency}&to={to_currency}",
        )
        data      = json.loads(response.data)
        rate      = data["rates"][to_currency]
        return {
            "amount":    amount,
            "from":      from_currency,
            "to":        to_currency,
            "rate":      rate,
            "converted": round(amount * rate, 2),
        }


# ── 1. Standalone usage ───────────────────────────────────────────────────────

tool   = CurrencyConverterTool()
result = tool.run(amount=100, from_currency="USD", to_currency="EUR")
print("[standalone]")
print(result)

# ── 2. In a Chain with a Skill ────────────────────────────────────────────────

format_skill = Skill(
    model = Model("claude-sonnet-4-6", api_key=os.getenv("ANTHROPIC_API_KEY")),
    input = {"messages": [{"role": "user", "parts": [
        "{amount} {from} = {converted} {to} (rate: {rate}). Format this as a friendly one-liner."
    ]}]},
)

chain  = Chain(steps=[tool, format_skill])
answer = chain.run(variables={"amount": 250, "from_currency": "GBP", "to_currency": "JPY"})
print("\n[chain]")
print(answer)
