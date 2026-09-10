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

from ..clients._base import APIError
from ..clients._errors import (
    RateLimitError, ServerError, NetworkError, TaskFailedError,
    InsufficientCreditsError,
)
from ..models._usage import Usage, extract_usage, attach_cost
from .._events import Event, emit
from . import _adapters as adapters

if TYPE_CHECKING:
    from ..models._base import Model

# HTTP status codes that indicate a transient server-side condition and are
# safe to retry.  Client errors (4xx other than 429) are permanent — they
# indicate a problem with the request itself and must not be retried.
_TRANSIENT_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# Output spec for intermediate generate turns in multi-turn directed reasoning —
# always plain text (they only feed the running context; only the final turn
# uses the skill's real output format).
_TEXT_OUTPUT: dict = {"modalities": ["text"], "format": {"type": "text"}}

# Error types that trigger fallback to the next model in the chain.  Auth /
# invalid-request / not-found are NOT here — falling back would hide them.
_FALLBACK_ERRORS = (RateLimitError, ServerError, NetworkError)


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

        **Multi-turn (directed reasoning, 1.5.0).** Several ``user`` turns
        separated by a "generate here" marker — an ``assistant`` turn with no
        ``parts`` — run as a sequence: each marker is one model call whose reply
        is appended to the running context (an ``assistant`` turn *with* parts
        is a fixed/seed turn, not a call). ``run()`` returns the last reply;
        ``history`` holds each turn. Backward compatible: no markers → one call.

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
        model:        "Model | list[Model]",
        input:        dict,
        output:       "dict | None" = None,
        variables:    dict  | None  = None,
        options:      dict  | None  = None,
        name:         str   | None  = None,
        description:  str   | None  = None,
        max_retries:  int           = 0,
        retry_delay:  float         = 2.0,
        hooks:        list  | None  = None,
        _tools:       list  | None  = None,
    ) -> None:
        input  = adapters.normalize_input(input)
        output = adapters.normalize_output(output)
        adapters.validate_input(input)
        adapters.validate_output(output)

        # A single model or a fallback chain: on a transient failure
        # (rate limit / server / network) the next model is tried.  A bare
        # model behaves exactly as before — self.model stays the primary.
        self.models = list(model) if isinstance(model, (list, tuple)) else [model]
        if not self.models:
            raise ValueError("Skill requires at least one model.")
        self.model       = self.models[0]
        self._input      = input
        self._output     = output
        # Internal seam, not public API — the leading underscore is the
        # contract. The Agent declares its tools here so the provider receives
        # them natively; Skill only *transports* declarations and returns the
        # model's calls, it never executes anything. Callers who reach for
        # this directly are walking on internals and have signed for it; the
        # public way to get a raw decision without execution is Agent.step().
        self._tools      = list(_tools) if _tools else None
        self.variables   = variables or {}
        self.options     = options   or {}
        self.name        = name
        self.description = description
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.hooks       = list(hooks or [])     # observability hooks (1.4.4)

        # Token usage of the most recent run() — None until the first call.
        # Reading it is optional; it never affects run()'s inputs or output.
        # For a multi-turn run it is the sum across turns.
        self.last_usage: "Usage | None" = None
        #: What the library had to change about the last call to fit the
        #: provider — empty when it went out as asked. A warning is easy to
        #: miss in a log; this is what a run record can keep, so a comparison
        #: between two models can state what its arms actually sent.
        self.last_adaptations: list = []
        # Generated replies of the most recent run(), in order (multi-turn
        # directed reasoning); ``history[-1]`` is the returned value. A single-
        # shot run leaves a one-element list. ``None`` until the first call.
        self.history: "list | None" = None

        #: The whole answer from the last :meth:`stream`, assembled and
        #: parsed. None until a stream completes.
        self.last_result: "str | dict | None" = None

    def _emit(self, etype: str, **fields) -> None:
        """Dispatch an observability :class:`Event` to registered hooks (1.4.4)."""
        if self.hooks:
            emit(self.hooks, Event(type=etype, **fields))

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
            ``{429, 500, 502, 503, 504}``).  With a fallback chain, the last
            model's transient error is raised only after every model failed.
        ValueError
            If the provider response cannot be parsed (never retried —
            this indicates a prompt or schema error, not a transient fault).
        """
        _max_retries = self.max_retries if max_retries is None else max_retries
        _retry_delay = self.retry_delay if retry_delay is None else retry_delay

        # Merge variables: instance defaults ← call-time overrides
        merged = {**self.variables, **(variables or {})}

        # Substitute {placeholders} in a deep copy of the messages list, then
        # read any file the parts point at. Order matters: the path is itself
        # a template, so one Skill can serve a whole Pool of images.
        messages = adapters.substitute(self._input["messages"], merged)
        messages = adapters.resolve_media(messages)

        # Reset usage/history so that, if this call fails, they are None rather
        # than a stale value left over from a previous successful run().
        self.last_usage = None
        self.last_adaptations = []
        self.history    = None

        # Try each model in the fallback chain.  Transient failures
        # (rate limit / server / network) advance to the next model; a
        # non-transient failure (bad request, auth, parse) propagates
        # immediately — falling back would only hide a real error.
        for i, model in enumerate(self.models):
            try:
                return self._run_on_model(
                    model, messages, _max_retries, _retry_delay
                )
            except _FALLBACK_ERRORS:
                if i < len(self.models) - 1:
                    continue   # try the next model in the chain
                raise          # last model exhausted

    def stream(self, variables: "dict | None" = None):
        """
        Run the skill and yield the answer as it arrives.

        ``run()`` is untouched — this is a second way to spend the same
        request, not a replacement — and the differences from it are real
        rather than incidental, so they are stated instead of discovered:

        * **No retries and no fallback chain.** Both work by throwing the
          attempt away and starting again, which a stream cannot do once the
          caller has seen the first piece. A failure before the first piece
          could be retried; a failure after it could not, so the rule would
          hold only sometimes, and a rule that holds sometimes is worse than
          none. The first model in the chain is used, and an error is raised.
        * **The result is assembled as well as yielded.** ``last_result``
          holds the whole text at the end, parsed when the output format
          asks for JSON, so a caller does not have to choose between showing
          progress and having the value.
        * **Usage is whatever the provider reported**, on ``last_usage``, and
          None when it reported nothing.

        A provider that cannot stream yields the whole answer as one piece
        and says so through the adaptation channel — see ``Model.stream``.
        """
        import json as _json
        import time as _time

        merged = {**self.variables, **(variables or {})}
        messages = adapters.substitute(self._input["messages"], merged)
        messages = adapters.resolve_media(messages)

        self.last_usage = None
        self.last_adaptations = []
        self.history = None
        self.last_result = None

        model  = self.models[0]
        _tools = getattr(self, "_tools", None)
        pieces = []

        self._emit("llm_call.started", name=model.name)
        _t0 = _time.monotonic()
        try:
            for piece in model.stream(messages, self._output, tools=_tools):
                pieces.append(piece)
                yield piece
        except Exception as exc:
            self._emit("llm_call.ended", name=model.name,
                       duration=_time.monotonic() - _t0, error=str(exc))
            raise

        self.last_adaptations = list(getattr(model, "last_adaptations", []))
        raw_usage = getattr(model, "last_stream_usage", None)
        if raw_usage:
            # `raw_usage` is already response-shaped — see
            # `BaseClient.stream_usage` — so it reaches the same branch of
            # `extract_usage` a buffered response does, per provider.
            self.last_usage = attach_cost(
                extract_usage(raw_usage), model.name,
                getattr(model, "cache_ttl", "5m"))

        text = "".join(pieces)
        kind = ((self._output or {}).get("format") or {}).get("type", "text")
        if kind in ("json", "json_schema"):
            try:
                self.last_result = _json.loads(text)
            except _json.JSONDecodeError:
                # The pieces are handed over as they came; a truncated or
                # non-JSON answer is not silently turned into None, because
                # the caller has already seen the text and can say what
                # arrived better than a swallowed exception can.
                self.last_result = text
        else:
            self.last_result = text
        self.history = [self.last_result]
        self._emit("llm_call.ended", name=model.name,
                   usage=getattr(self.last_usage, "total_tokens", None),
                   cost=getattr(self.last_usage, "cost", None),
                   duration=_time.monotonic() - _t0)

    def _run_on_model(
        self,
        model:       "Model",
        messages:    list,
        max_retries: int,
        retry_delay: float,
    ) -> "str | dict":
        """
        Run *messages* against a single model.

        Single-shot when there are no ``assistant`` generate markers (the
        original behavior). With markers (multi-turn directed reasoning), each
        marker — and the implicit final generate after a trailing ``user`` turn
        — is one model call; each reply is appended to the running context so
        later turns see it. The last call uses the skill's real output format;
        intermediate calls are plain text (they only feed the context).
        ``last_usage`` is the sum across turns; ``history`` holds each reply.
        """
        if not any(adapters.is_generate_marker(m) for m in messages):
            result, usage = self._call_once(
                model, messages, self._output, max_retries, retry_delay)
            self.last_usage = usage
            # A Model-like object from outside the library need not have
            # grown this attribute; a new field of ours must not break code
            # that was working.
            self.last_adaptations = list(
                getattr(model, "last_adaptations", ()) or ())
            self.history    = [result]
            return result

        # Build the turn sequence: real messages interleaved with generate
        # points; add an implicit final generate when the script ends on an
        # unanswered ``user`` turn.
        seq: list = [("gen",) if adapters.is_generate_marker(m) else ("msg", m)
                     for m in messages]
        if seq and seq[-1][0] == "msg" and seq[-1][1].get("role") == "user":
            seq.append(("gen",))
        last_gen = max(k for k, s in enumerate(seq) if s[0] == "gen")

        running: list = []
        total_usage = None
        history: list = []
        final: "str | dict | None" = None
        for k, item in enumerate(seq):
            if item[0] == "msg":
                running.append(item[1])
                continue
            output = self._output if k == last_gen else _TEXT_OUTPUT
            reply, usage = self._call_once(
                model, running, output, max_retries, retry_delay)
            total_usage = usage if total_usage is None else total_usage + usage
            history.append(reply)
            final = reply
            # Splice the reply into the context as an assistant turn so the next
            # turn sees it. A non-text (json) reply is stringified for context.
            text = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
            running.append({"role": "assistant",
                            "parts": [{"type": "text", "text": text}]})

        self.last_usage = total_usage
        self.last_adaptations = list(
            getattr(model, "last_adaptations", ()) or ())
        self.history    = history
        return final

    def _call_once(
        self,
        model:       "Model",
        messages:    list,
        output:      dict,
        max_retries: int,
        retry_delay: float,
    ) -> "tuple[str | dict, Usage]":
        """One model call with transient retries; returns ``(result, usage)``."""
        # Pass tools only when declared: this boundary accepts anything
        # model-shaped (tests stub bare duck types), and a kwarg the double
        # never heard of would break every caller that wants no tools at all.
        _tools = getattr(self, "_tools", None)
        if _tools:
            path, body = model.to_request(messages, output, tools=_tools)
        else:
            path, body = model.to_request(messages, output)

        for attempt in range(max(0, max_retries) + 1):
            if attempt > 0:
                time.sleep(retry_delay * (2 ** (attempt - 1)))

            try:
                self._emit("llm_call.started", name=model.name)
                _t0 = time.monotonic()
                raw      = model.client.send(
                    path, body, model.client._auth_headers()
                )
                response = json.loads(raw)
                # ``cache_ttl`` is read leniently where ``name`` is not: a
                # Model always has it, but this boundary accepts anything
                # model-shaped, and the lifetime only picks a price multiplier.
                # A cost estimate is the last thing that should abort a call
                # that already succeeded.
                usage    = attach_cost(extract_usage(response), model.name,
                                       getattr(model, "cache_ttl", "5m"))
                result   = model.from_response(response, output)
                self._emit("llm_call.ended", name=model.name,
                           usage=getattr(usage, "total_tokens", None),
                           cost=getattr(usage, "cost", None),
                           duration=time.monotonic() - _t0)
                return result, usage

            except APIError as exc:
                self._emit("llm_call.ended", name=model.name,
                           duration=time.monotonic() - _t0, error=str(exc))
                # Retry transient server conditions and network failures
                # (NetworkError has status 0, so check the type too). Never
                # retry TaskFailedError (would create a new billable async job)
                # or InsufficientCreditsError (which can arrive as 429 but is
                # terminal — retrying the same account won't add funds).
                transient = (exc.status in _TRANSIENT_STATUSES
                             or isinstance(exc, NetworkError))
                if (transient
                        and not isinstance(exc, (TaskFailedError,
                                                 InsufficientCreditsError))
                        and attempt < max_retries):
                    continue   # wait and retry the same model
                raise          # non-transient, or retries exhausted

        # Unreachable: the loop runs at least once (max(0, max_retries)+1 >= 1)
        # and every iteration returns or raises.
        raise RuntimeError("retry loop exited without returning or raising")

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

        from ..models._base import Model as _Model
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
