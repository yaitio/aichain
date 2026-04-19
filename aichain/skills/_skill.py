"""
skills._skill
=============

``Skill`` — the universal task runner.

A Skill binds a :class:`~models.Model` to:

  * a provider-agnostic input template (messages with ``{placeholder}``
    variables)
  * a declared output format (``text`` / ``json`` / ``json_schema``)
  * an optional set of default variable values

Calling ``skill.run()`` substitutes variables, delegates serialisation to
the model's own ``to_request()`` method, executes the HTTP request via the
model's attached client, then delegates deserialisation to ``from_response()``.

Skills can be persisted to YAML with :meth:`Skill.save` and reloaded with
the :meth:`Skill.load` class method.
"""

import json
import os
import time
from typing import TYPE_CHECKING

from clients._base import APIError
from . import _adapters as adapters

if TYPE_CHECKING:
    from models._base import Model

# HTTP status codes that indicate a transient server-side condition and are
# safe to retry.  Client errors (4xx other than 429) are permanent — they
# indicate a problem with the request itself and must not be retried.
_TRANSIENT_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


class Skill:
    """
    A reusable, model-bound task unit.

    Parameters
    ----------
    model : Model subclass instance
        The language model that will execute this skill.

    input : dict
        Universal message template.  Text parts may contain
        ``{placeholder}`` tokens that are filled in at run time.

        Shorthand — ``"type": "text"`` is the default; plain strings work too::

            {
              "messages": [
                {"role": "system", "parts": ["Be concise."]},
                {"role": "user",   "parts": [{"text": "Explain {topic}."}]},
              ]
            }

        Full explicit form (required only for non-text parts such as images)::

            {
              "messages": [
                {"role": "system", "parts": [{"type": "text", "text": "Be concise."}]},
                {"role": "user",   "parts": [{"type": "text", "text": "Explain {topic}."}]},
              ]
            }

    output : dict | None, optional
        Declares the expected output format.  Omit entirely (or pass ``None``
        / ``{}``) for the default plain-text output.

        Defaults to ``{"modalities": ["text"], "format": {"type": "text"}}``
        when omitted.  Only specify this when deviating from the default::

            # plain text — these are all equivalent:
            output=None
            output={}
            output={"modalities": ["text"], "format": {"type": "text"}}

            # JSON object
            output={"format": {"type": "json"}}

            # validated JSON schema
            output={"format": {"type": "json_schema", "schema": { ... }}}

            # image generation
            output={"modalities": ["image"], "format": {"type": "image"}}

    variables : dict | None, optional
        Default variable values.  Can be overridden or extended at call
        time via ``run(variables={...})``.  Merged at run time; call-time
        values take precedence.

    options : dict | None, optional
        Reserved for future use (currently stored but not applied).

    name : str | None, optional
        Human-readable label for the skill.

    description : str | None, optional
        Short description of what the skill does.

    Examples
    --------
    Text output (minimal — ``output`` omitted, shorthand parts)::

        from models import Model
        from skills import Skill

        skill = Skill(
            model=Model("gpt-4o"),
            input={
                "messages": [
                    {"role": "system", "parts": ["Be concise."]},
                    {"role": "user",   "parts": [{"text": "What is {topic}?"}]},
                ]
            },
            variables={"topic": "gravity"},
            name="explainer",
        )
        result = skill.run()                                  # uses default variable
        result = skill.run(variables={"topic": "relativity"}) # override

    JSON output::

        skill = Skill(
            model=Model("claude-sonnet-4-5"),
            input={
                "messages": [
                    {"role": "user", "parts": [{"text": "List 3 facts about {topic} as JSON."}]},
                ]
            },
            output={"format": {"type": "json"}},
            variables={"topic": "Mars"},
        )
        data = skill.run()   # returns dict

    Save and load::

        skill.save("skills/explainer.yaml")
        loaded = Skill.load("skills/explainer.yaml")   # model re-created from YAML
    """

    def __init__(
        self,
        model:        "Model",
        input:        dict,
        output:       "dict | None" = None,
        variables:    dict  | None  = None,
        options:      dict  | None  = None,
        name:         str   | None  = None,
        description:  str   | None  = None,
        max_retries:  int           = 0,
        retry_delay:  float         = 2.0,
    ) -> None:
        input  = adapters.normalize_input(input)
        output = adapters.normalize_output(output)
        adapters.validate_input(input)
        adapters.validate_output(output)

        self.model       = model
        self._input      = input
        self._output     = output
        self.variables   = variables or {}
        self.options     = options   or {}
        self.name        = name
        self.description = description
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(
        self,
        variables:   dict  | None = None,
        max_retries: int   | None = None,
        retry_delay: float | None = None,
    ) -> "str | dict":
        """
        Execute the skill and return the model's response.

        Parameters
        ----------
        variables : dict | None, optional
            Variables to substitute in this invocation.  Merged with the
            instance-level ``self.variables``; call-time values win.
        max_retries : int | None, optional
            Application-level retry count for this call only.  Overrides the
            constructor value when provided.  ``0`` means a single attempt
            with no retries (the default).
        retry_delay : float | None, optional
            Base sleep in seconds between retries for this call only.
            Overrides the constructor value when provided.  Each subsequent
            attempt doubles the delay (exponential back-off).

        Returns
        -------
        str
            When ``output["format"]["type"] == "text"``.
        dict
            When ``output["format"]["type"]`` is ``"json"`` or
            ``"json_schema"``.

        Raises
        ------
        clients._base.APIError
            On a non-transient HTTP error, or after all retry attempts are
            exhausted on a transient error (status in
            ``{429, 500, 502, 503, 504}``).
        ValueError
            If the provider response cannot be parsed (never retried —
            this indicates a prompt or schema error, not a transient fault).
        """
        _max_retries = self.max_retries if max_retries is None else max_retries
        _retry_delay = self.retry_delay if retry_delay is None else retry_delay

        # Merge variables: instance defaults ← call-time overrides
        merged = {**self.variables, **(variables or {})}

        # Substitute {placeholders} in a deep copy of the messages list
        messages = adapters.substitute(self._input["messages"], merged)

        # Build the native (path, body) pair once — the payload is the same
        # on every attempt; only the server-side condition changes.
        path, body = self.model.to_request(messages, self._output)

        last_error: "APIError | None" = None

        for attempt in range(_max_retries + 1):
            if attempt > 0:
                time.sleep(_retry_delay * (2 ** (attempt - 1)))

            try:
                raw      = self.model.client._post(
                    path, body, self.model.client._auth_headers()
                )
                response = json.loads(raw)
                return self.model.from_response(response, self._output)

            except APIError as exc:
                if exc.status in _TRANSIENT_STATUSES and attempt < _max_retries:
                    last_error = exc
                    continue   # wait and retry
                raise          # non-transient, or retries exhausted

        raise last_error  # type: ignore[misc]  # unreachable; satisfies linters

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serialise this skill to a YAML file at *path*.

        The file stores everything needed to recreate the skill — the model
        name, input template, output spec, variables, options, name, and
        description.  API keys are **never** written; :meth:`load` resolves
        them from environment variables at load time.

        Parameters
        ----------
        path : str
            Destination file path.  Parent directories are created
            automatically if they do not exist.

        Example
        -------
        ::

            skill.save("skills/translation.yaml")
        """
        try:
            import yaml  # PyYAML — optional dependency
        except ImportError:
            raise ImportError(
                "PyYAML is required for Skill.save(). "
                "Install it with: pip install pyyaml"
            )

        data = {
            "model_name":  self.model.name,
            "name":        self.name,
            "description": self.description,
            "input":       self._input,
            "output":      self._output,
            "variables":   self.variables if self.variables else None,
            "options":     self.options   if self.options   else None,
        }
        # Drop None values to keep the YAML tidy
        data = {k: v for k, v in data.items() if v is not None}

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(data, fh, allow_unicode=True, sort_keys=False,
                      default_flow_style=False)

    @classmethod
    def load(cls, path: str, api_key: "str | None" = None) -> "Skill":
        """
        Load a skill from a YAML file previously created by :meth:`save`.

        The model is constructed automatically from the ``model_name`` stored
        in the file.  The API key is resolved from the matching environment
        variable (e.g. ``OPENAI_API_KEY``) unless *api_key* is supplied
        explicitly.

        Parameters
        ----------
        path    : str
            Path to the YAML file.
        api_key : str | None, optional
            Override the API key used to initialise the model.  When omitted,
            the key is read from the provider's environment variable.

        Returns
        -------
        Skill
            A fully initialised :class:`Skill` instance with a ready-to-use
            model attached.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ImportError
            If PyYAML is not installed.
        ValueError
            If no API key is available for the model's provider.

        Example
        -------
        ::

            from skills import Skill

            skill  = Skill.load("skills/translation.yaml")
            result = skill.run(variables={"text": "Hello, world!"})
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for Skill.load(). "
                "Install it with: pip install pyyaml"
            )

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        from models._base import Model as _Model
        model = _Model(data["model_name"], api_key=api_key)

        return cls(
            model       = model,
            input       = data["input"],
            output      = data["output"],
            variables   = data.get("variables"),
            options     = data.get("options"),
            name        = data.get("name"),
            description = data.get("description"),
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [f"model={self.model.name!r}"]
        if self.name:
            parts.append(f"name={self.name!r}")
        if self.description:
            parts.append(f"description={self.description!r}")
        if self.variables:
            parts.append(f"variables={self.variables!r}")
        return f"Skill({', '.join(parts)})"
