"""
18_rest.py — Your own API as tools, without code per method; the risk class follows the method.

Keys: none; describes the free Frankfurter exchange-rate API.
"""

from yait_aichain.utilities import REST

rates = REST("https://api.frankfurter.dev/v1", description="Currency exchange rates",
             headers={"User-Agent": "aichain-example"})
latest = rates.endpoint("latest", "GET", "/latest", description="Today's rates from one currency",
                        query_params=["from", "to"], fields={"from": (str, "base currency, e.g. USD"),
                                                             "to": (str, "target currencies, comma separated")})
currencies = rates.endpoint("currencies", "GET", "/currencies",
                            description="Every currency the API knows, by code")
save = rates.endpoint("save_alert", "POST", "/alerts", description="Store a rate alert",
                      body_params=["pair", "below"])         # an endpoint that changes something

for t in (latest, currencies, save):
    print(f"{t.name}  {t.risk}")

print(latest.run({"from": "USD", "to": "EUR,GBP"}))
# An agent takes them as tools: Agent(model, tools=[latest, currencies]); a policy reads t.risk.
