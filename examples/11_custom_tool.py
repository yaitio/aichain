"""
11_custom_tool.py — Any function becomes a tool: its docstring is the schema a model sees.

Keys: none; calls the free Frankfurter exchange-rate API.
"""

import json
import urllib.request

from yait_aichain.tools import tool


@tool(risk="read")
def currency_converter(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert an amount between two currencies at today's rate.

    Parameters
    ----------
    amount : float
        How much to convert.
    from_currency : str
        The currency it is in, as a three-letter code: USD, EUR, GBP.
    to_currency : str
        The currency to convert it to.
    """
    url = (f"https://api.frankfurter.dev/v1/latest?amount={amount}"
           f"&from={from_currency.upper()}&to={to_currency.upper()}")
    request = urllib.request.Request(url, headers={"User-Agent": "aichain-example"})
    with urllib.request.urlopen(request, timeout=10) as response:
        rates = json.load(response)["rates"]
    return {"converted": rates[to_currency.upper()], "currency": to_currency.upper()}


print(json.dumps(currency_converter.schema(), indent=2))
result = currency_converter.run(amount=100, from_currency="USD", to_currency="EUR")
print(f"converted: {result['converted']} {result['currency']}")
