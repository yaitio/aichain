"""
utilities.rest
==============

``REST`` — an HTTP API as a utility. The base URL, a description of the API,
the authentication and the headers are set once for the whole API; each
endpoint is a tool.

::

    api = REST("https://restful-booker.herokuapp.com",
               description="Restful Booker, a hotel booking API",
               headers={"Accept": "application/json"})
    get_booking = api.endpoint("get_booking", "GET", "/booking/{id}", path_params=["id"],
                               description="One booking by its id.")
    get_booking.run(id=1)
    Agent(model, tools=api.tools)                    # every endpoint made so far

Each endpoint is a ``RestApiTool``, which can also be built on its own when
an API has a single endpoint. Its risk class follows the method — ``GET`` is
``read``, ``DELETE`` is ``destructive``, a write is ``external`` — unless
``risk=`` says otherwise.
"""

from __future__ import annotations

from .._base import Utility
from ._tool import RestApiTool


class REST(Utility):
    """
    An HTTP API: ``endpoint(...)`` makes each of its tools; ``tools`` lists them.

    ``description`` says what the API is, and is required: an endpoint's own
    description — "One booking by its id." — does not tell a model which
    system it is in, so every endpoint carries the API's description before its
    own: "Restful Booker, a hotel booking API — One booking by its id."
    """

    def __init__(self, base_url: str, *, description: str, auth: "dict | None" = None,
                 headers: "dict | None" = None, timeout: "float | None" = None) -> None:
        if not isinstance(base_url, str) or not base_url.lower().startswith(("http://", "https://")):
            raise ValueError(f"REST needs an http or https base URL; got {base_url!r}")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("REST needs a description of the API — a model decides whether to call an "
                             "endpoint by it, and an endpoint alone does not say which system it is in")
        self.base_url = base_url.rstrip("/")
        self.description = description.strip()
        self.auth = dict(auth or {})
        self.headers = dict(headers or {})
        self.timeout = timeout
        self._endpoints: dict = {}

    def endpoint(self, name: str, method: str, path: str, *, description: str,
                 path_params=(), query_params=(), body_params=(), required_params=(),
                 content_type: str = "application/json", response_field: "str | None" = None,
                 risk: "str | None" = None) -> RestApiTool:
        """
        A tool for one endpoint: *method* on the base URL plus *path* —
        ``"/booking/{id}"`` with ``path_params=["id"]``. Its fields are its
        arguments; the API's auth and headers are the utility's.

        *risk* is what a permission policy reads; omitted, it follows the
        method — ``GET`` is ``read``, ``DELETE`` is ``destructive``,
        ``POST``/``PUT``/``PATCH`` are ``external``.
        """
        if name in self._endpoints:
            raise ValueError(f"this API already has an endpoint named {name!r}")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"endpoint {name!r} needs a description — what it does, for the model")
        tool = RestApiTool(name, f"{self.description.rstrip('.')} — {description.strip()}", method,
                           self.base_url + "/" + path.lstrip("/"),
                           path_params=path_params, query_params=query_params, body_params=body_params,
                           required_params=required_params, headers=self.headers, auth=self.auth,
                           content_type=content_type, response_field=response_field,
                           timeout=self.timeout, risk=risk)
        self._endpoints[name] = tool
        return tool

    @property
    def TOOLS(self) -> tuple:
        return tuple(self._endpoints)

    @property
    def tools(self) -> list:
        return list(self._endpoints.values())

    def __getattr__(self, name):
        # api.get_booking once made — the endpoint as an attribute, like a utility's other tools.
        endpoints = self.__dict__.get("_endpoints", {})
        if name in endpoints:
            return endpoints[name]
        raise AttributeError(f"{type(self).__name__} has no attribute or endpoint {name!r}")

    def __repr__(self) -> str:
        return f"REST({self.base_url!r}, description={self.description!r}, endpoints={list(self._endpoints)})"


__all__ = ["REST", "RestApiTool"]
