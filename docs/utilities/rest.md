## `REST` — an API as a utility

```python
from yait_aichain.utilities import REST

api = REST("https://restful-booker.herokuapp.com", description="Restful Booker, a hotel booking API",
           headers={"Accept": "application/json"})
get_booking = api.endpoint("get_booking", "GET", "/booking/{id}", path_params=["id"],
                           description="One booking by its id.")
get_booking.run(id=1)
Agent(model, tools=api.tools)                        # every endpoint made so far
```

The base URL, `description`, `auth` and `headers` are the API's, set once and
shared by every endpoint. `description` is required: an endpoint's own — "One
booking by its id." — does not tell a model which system it is in, so every
endpoint carries the API's before its own: "Restful Booker, a hotel booking API —
One booking by its id." An endpoint is a `RestApiTool`, named and described for the model, and
reachable as `api.get_booking` once made.
