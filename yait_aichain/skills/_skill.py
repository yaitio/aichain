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


from .._budget import as_budget, in_force, start_run
import json
import random
import time
from typing import TYPE_CHECKING

from .. import _hints
from ..clients._base import APIError
from ..clients._errors import (
    RateLimitError, ServerError, NetworkError, TaskFailedError,
    InsufficientCreditsError, InvalidStructuredOutputError,
)
from ..models._adaptation import SWAPPED, Adaptation
from ..models._base import Model, StreamOutcome
from ..models._calls import ToolCallRequest
from ..models._schema import check_structure
from ..clients._families._structured import parse_structured
from ..models._usage import Usage, extract_usage, attach_cost
from .._events import Event, emit
from . import _adapters as adapters

if TYPE_CHECKING:
    from .._budget import Budget

#: What a retry or a fallback may answer: a rate limit, a server error —
#: Anthropic's 529 overloaded among them — or no response at all. One tuple
#: for both, where a list of statuses for the retry and a list of classes for
#: the fallback had already drifted apart.
_TRANSIENT = (RateLimitError, ServerError, NetworkError)

#: The longest pause between two attempts, whatever the server asks for.
_MAX_BACKOFF = 60.0


def _attempts(max_retries: int, delay: float):
    """Attempt numbers from 0; the pause before each retry is set by `_pause`."""
    if delay < 0:
        raise ValueError(f"retry_delay is seconds, at least 0; got {delay!r}")
    yield from range(max(0, max_retries) + 1)


def _pause(exc: APIError, attempt: int, delay: float) -> None:
    """Wait before attempt *attempt* + 1: twice as long each time, at least as
    long as the server's ``Retry-After``, with up to a second of jitter, and
    never past `_MAX_BACKOFF`."""
    wait = max(getattr(exc, "retry_after", None) or 0.0, delay * (2 ** attempt))
    time.sleep(min(_MAX_BACKOFF, wait + random.uniform(0, 1.0 if wait else 0.0)))


def _tee(stream, into: list):
    """Pass a stream's pieces through, keeping them in *into*, and return
    what the stream returned when it ended."""
    while True:
        try:
            piece = next(stream)
        except StopIteration as done:
            return done.value
        into.append(piece)
        yield piece


def _retryable(exc: APIError, attempt: int, max_retries: int) -> bool:
    """A transient failure with retries left. Never a failed async task, whose
    retry is a new billable job, nor an empty balance, which can arrive as 429
    and which no retry on the same account refills."""
    return (isinstance(exc, _TRANSIENT)
            and not isinstance(exc, (TaskFailedError, InsufficientCreditsError))
            and attempt < max_retries)


# Output spec for intermediate generate turns in multi-turn directed reasoning —
# always plain text (they only feed the running context; only the final turn
# uses the skill's real output format).
_TEXT_OUTPUT: dict = {"modalities": ["text"], "format": {"type": "text"}}



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

        **Multi-turn (directed reasoning).** Several ``user`` turns
        separated by a "generate here" marker — an ``assistant`` turn with no
        ``parts`` — run as a sequence: each marker is one model call whose reply
        is appended to the running context (an ``assistant`` turn *with* parts
        is a fixed/seed turn, not a call). ``run()`` returns the last reply;
        ``history`` holds each turn. No markers → one call.

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

        A Skill makes text: ``output={"modalities": ["image"]}`` raises
        ``ValueError``. Images are a tool — ``image(service, model=…)``.

    variables : dict | None, optional
        Default variable values.  Can be overridden or extended at call
        time via ``run(variables={...})``.  Merged at run time; call-time
        values take precedence.

    **refused
        ``options=`` is refused: it was stored and never applied, so
        ``Skill(options={"temperature": 0})`` ran at the model's default. A
        model's options are ``Model(name, options={…})``.

    name : str | None, optional
        Human-readable label for the skill.

    description : str | None, optional
        Short description of what the skill does.
    prompt : str | None, optional
        A shortcut for one user message: ``Skill(model, prompt="Summarise:
        {text}")``. Give ``prompt`` or ``input``, not both.
    max_retries : int, optional
        Retries after a transient failure (429, 5xx, a network error), on top
        of the transport's own. ``0``, the default, is one attempt.
    retry_delay : float, optional
        Seconds before the first retry; each later one waits twice as long.
    hooks : list | None, optional
        Callables given an ``Event`` for each model call.
    max_cost : float | Budget | None, optional
        A ceiling in dollars; a call that would begin past it raises
        ``BudgetExceeded``.

    Examples
    --------
    Text output (minimal — ``output`` omitted, shorthand parts)::

        from yait_aichain.models import Model
        from yait_aichain.skills import Skill

        skill = Skill(
            model=Model("gpt-6-luna"),
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
            model=Model("claude-haiku-5-5"),
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

    @_hints.refuse("Skill")
    def __init__(
        self,
        model:        "Model | list[Model]",
        input:        "dict | None" = None,
        *,
        prompt:       "str | None" = None,
        output:       "dict | None" = None,
        variables:    dict  | None  = None,
        name:         str   | None  = None,
        description:  str   | None  = None,
        max_retries:  int           = 0,
        retry_delay:  float         = 2.0,
        hooks:        list  | None  = None,
        max_cost:     "float | Budget | None" = None,
        _tools:       list  | None  = None,
        _image:       bool          = False,
        **refused,
    ) -> None:
        if refused:
            raise TypeError(
                "Skill takes no " + ", ".join(f"{k}=" for k in sorted(refused))
                + (" — a model's options are Model(name, options={…}); Skill(options=…) was "
                   "stored and never applied" if "options" in refused else ""))
        # `prompt=` is the shortcut for the commonest shape there is: one
        # user message, one text part. The long form stays the only way to say
        # anything else — a system message, several turns, an image — because
        # a shortcut that grows options becomes a second input format, and two
        # ways to say the same thing is how the docs and the code drift apart.
        if prompt is not None:
            if input is not None:
                raise ValueError(
                    "pass prompt= or input=, not both: prompt is shorthand "
                    "for input={'messages': [{'role': 'user', 'parts': "
                    "[prompt]}]}, so giving both leaves it ambiguous which "
                    "one the model should see.")
            input = {"messages": [{"role": "user", "parts": [prompt]}]}
        if input is None:
            raise ValueError(
                "Skill needs input= (the full messages spec) or prompt= (one "
                "user message).")
        input  = adapters.normalize_input(input)
        output = adapters.normalize_output(output)
        adapters.validate_input(input)
        adapters.validate_output(output)

        # A single model or a fallback chain: on a transient failure
        # (rate limit / server / network) the next model is tried.  A bare
        # model is a chain of one — self.model is always the primary.
        self.models = list(model) if isinstance(model, (list, tuple)) else [model]
        if not self.models:
            raise ValueError("Skill requires at least one model.")
        named = [m for m in self.models if isinstance(m, str)]
        if named:
            raise TypeError(f"Skill takes a Model, not its name: Skill(Model({named[0]!r}), ...)")
        self.model       = self.models[0]
        self._input      = input
        self._output     = output
        wants_image = output and ("image" in (output.get("modalities") or [])
                                  or (output.get("format") or {}).get("type") == "image")
        if wants_image and not _image:
            # An image model is a tool: it cannot be an agent's mind. The image
            # tool builds its request through a Skill, by the `_image` seam,
            # and that seam is the tool's alone.
            raise ValueError(
                "a Skill makes text; an image model is a tool — "
                "image(service, model=…).run(prompt, options={…}), or "
                "images=[…] to edit")
        # Internal seam, not public API — the leading underscore is the
        # contract. The Agent declares its tools here so the provider receives
        # them natively; Skill only *transports* declarations and returns the
        # model's calls, it never executes anything. Callers who reach for
        # this directly are walking on internals and have signed for it; the
        # public way to get a raw decision without execution is Agent.step().
        self._tools      = list(_tools) if _tools else None
        self.variables   = variables or {}
        self.name        = name
        self.description = description
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.hooks       = list(hooks or [])     # observability hooks

        # Token usage of the most recent run() — None until the first call.
        # Reading it is optional; it never affects run()'s inputs or output.
        # For a multi-turn run it is the sum across turns.
        # A ceiling in money. Takes a number or a shared `Budget`, because a
        # chain hands the same object to every step: separate copies would let
        # each level spend the full amount, which is the failure this exists
        # to prevent.
        self.max_cost = as_budget(max_cost)
        self._own_budget = self.max_cost        # see _budget.start_run

        self.last_usage: Usage | None = None
        #: What the library had to change about the last call to fit the
        #: provider — empty when it went out as asked. A warning is easy to
        #: miss in a log; this is what a run record can keep, so a comparison
        #: between two models can state what its arms actually sent.
        self.last_adaptations: list = []
        # Generated replies of the most recent run(), in order (multi-turn
        # directed reasoning); ``history[-1]`` is the returned value. A single-
        # shot run leaves a one-element list. ``None`` until the first call.
        self.history: list | None = None

        #: The whole answer from the last :meth:`stream`, assembled and
        #: parsed. None until a stream completes.
        self.last_result: str | dict | None = None

    def _emit(self, etype: str, **fields) -> None:
        """Dispatch an observability :class:`Event` to registered hooks."""
        if self.hooks:
            emit(self.hooks, Event(type=etype, **fields))

    def _started(self, model) -> float:
        """Check the budget, announce the call, and return its start time.

        Before the call, not after: the length of a reply is not known until
        it is paid for, so a budget bounds "do not begin another call" and
        never "do not exceed by a cent".
        """
        for budget in in_force(self.max_cost):
            budget.check(f"{model.name} for skill {self.name or '(unnamed)'}")
        self._emit("llm_call.started", name=model.name)
        return time.monotonic()

    def _ended(self, model, t0: float, usage=None, error=None) -> None:
        self._emit("llm_call.ended", name=model.name,
                   usage=getattr(usage, "total_tokens", None),
                   cost=getattr(usage, "cost", None),
                   duration=time.monotonic() - t0,
                   **({"error": str(error)} if error is not None else {}))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def __getattr__(self, name):
        raise _hints.missing(self, name)

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
        start_run(self)
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
        # The models that failed before the one that answered are recorded as
        # a swap: the answer, and its price, came from a different model than
        # the one asked for, and a run record has to be able to say so.
        failed: list = []
        for i, model in enumerate(self.models):
            try:
                result = self._run_on_model(
                    model, messages, _max_retries, _retry_delay
                )
            except _TRANSIENT as exc:
                if i == len(self.models) - 1:
                    for note in failed:           # the earlier failures, not lost
                        getattr(exc, "add_note", lambda _: None)(note)
                    raise      # last model exhausted
                failed.append(f"{model.name}: {exc}")
                continue       # try the next model in the chain
            if failed:
                self.last_adaptations.append(Adaptation(
                    SWAPPED, "model", asked=self.models[0].name, sent=model.name,
                    why="fallback after " + "; ".join(failed), model=self.models[0].name))
            return result

    def stream(self, variables: "dict | None" = None):
        """
        Run the skill and yield the answer as it arrives.

        ``run()`` is untouched — this is a second way to spend the same
        request, not a replacement — and the differences from it are real
        rather than incidental, so they are stated instead of discovered:

        * **Retries stop at the first piece; there is no fallback chain.**
          Both work by throwing the attempt away and starting again, which is
          impossible once the caller has seen something. *Before the first
          piece* is a state the caller can see — nothing has been yielded —
          so a transient failure (429, 5xx, network) retries
          while the attempt is still discardable, exactly as ``run()`` does,
          and is raised once it is not. The fallback chain does stay out: a
          second model is a different answer, not a retry of this one, and
          swapping models mid-sentence is not something to do quietly.
        * **The result is assembled as well as yielded.** ``last_result``
          holds the whole text at the end, parsed when the output format
          asks for JSON, so a caller does not have to choose between showing
          progress and having the value.
        * **Usage is whatever the provider reported**, on ``last_usage``, and
          None when it reported nothing.

        A provider that cannot stream yields the whole answer as one piece
        and says so through the adaptation channel — see ``Model.stream``.

        Parameters
        ----------
        variables : dict | None, optional
            Variables for this call, merged over the Skill's own.
        """
        start_run(self)
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
        finished = False
        try:
            yield from self._stream_on(model, messages, _tools, pieces)
            finished = True
        finally:
            # A caller that stops reading — a closed connection, a `break` —
            # ends the generator with GeneratorExit, which skipped the charge
            # and the ended event: a cancelled stream was never recorded.
            if not finished and self._open_call is not None:
                self._ended(model, self._open_call, error="abandoned")
            self._open_call = None

    _open_call: "float | None" = None

    def _stream_on(self, model, messages, _tools, pieces):
        outcome = None
        for attempt in _attempts(self.max_retries, self.retry_delay):
            t0 = self._started(model)
            self._open_call = t0
            try:
                # The outcome is this stream's own, returned when it ends — not
                # read back off a model another stream may be using.
                outcome = yield from _tee(model.stream(messages, self._output, tools=_tools), pieces)
                break
            except Exception as exc:
                self._open_call = None
                self._ended(model, t0, error=exc)
                # `pieces` is the whole condition that matters: once the
                # caller has seen something, starting over would replay or
                # contradict it. Before that, the attempt is discardable and
                # there is no reason to be less reliable than `run()`.
                if isinstance(exc, APIError) and not pieces and _retryable(exc, attempt, self.max_retries):
                    _pause(exc, attempt, self.retry_delay)
                    continue
                raise

        if outcome is None:                      # a model-shaped double that returns nothing
            outcome = StreamOutcome(getattr(model, "last_stream_result", None),
                                    getattr(model, "last_stream_usage", None),
                                    list(getattr(model, "last_adaptations", []) or []))
        self.last_adaptations = list(outcome.adaptations)
        raw_usage = outcome.usage
        if raw_usage:
            # `raw_usage` is already response-shaped — see
            # `BaseClient.stream_usage` — so it reaches the same branch of
            # `extract_usage` a buffered response does, per provider.
            self.last_usage = attach_cost(
                extract_usage(raw_usage), model.name,
                getattr(model, "cache_ttl", "5m"),
                getattr(model, "_provider", None))
            for budget in in_force(self.max_cost):
                budget.charge(getattr(self.last_usage, "cost", None))

        self._open_call = None
        fmt = (self._output or {}).get("format") or {}
        kind = fmt.get("type", "text")
        whole = outcome.result
        try:
            if isinstance(whole, ToolCallRequest) and kind == "json_schema" and not _tools:
                # Anthropic's structured output is a forced call of a tool the
                # library made up; its arguments are the answer, as run() has it.
                whole = self._structured_call(whole, fmt)
            if whole is not None and not isinstance(whole, str):
                # The provider could not stream and answered in one piece; the
                # object it answered with is the result, not the text it renders
                # to. A tool call rendered to text and parsed back is not a tool
                # call any more.
                self.last_result = whole
            elif kind in ("json", "json_schema"):
                # Parsed the way run() parses it — fences stripped, the schema
                # checked — and a failure raised, where it handed back the text.
                self.last_result = "".join(pieces)
                self.last_result = parse_structured(
                    self.last_result, fmt.get("schema") if kind == "json_schema" else None,
                    strip_fences=True)
            else:
                self.last_result = "".join(pieces)
        except Exception as exc:
            self._ended(model, t0, self.last_usage, error=exc)
            raise
        self.history = [self.last_result]
        self._ended(model, t0, self.last_usage)

    @staticmethod
    def _structured_call(request, fmt: dict):
        name = fmt.get("name", "structured_output")
        calls = [c for c in request.calls if c.name == name]
        if len(calls) != 1:
            return request
        value = calls[0].arguments
        if isinstance(fmt.get("schema"), dict) and (problems := check_structure(value, fmt["schema"])):
            raise InvalidStructuredOutputError(
                "Response does not match the requested schema: " + "; ".join(problems[:5]),
                problems=problems, value=value)
        return value

    def _run_on_model(
        self,
        model:       "Model",
        messages:    list,
        max_retries: int,
        retry_delay: float,
    ) -> "str | dict":
        """
        Run *messages* against a single model.

        Single-shot when there are no ``assistant`` generate markers. With
        markers (multi-turn directed reasoning), each
        marker — and the implicit final generate after a trailing ``user`` turn
        — is one model call; each reply is appended to the running context so
        later turns see it. The last call uses the skill's real output format;
        intermediate calls are plain text (they only feed the context).
        ``last_usage`` is the sum across turns; ``history`` holds each reply.
        """
        if not any(adapters.is_generate_marker(m) for m in messages):
            try:
                result, usage, made = self._call_once(
                    model, messages, self._output, max_retries, retry_delay)
            except Exception as exc:
                self.last_usage = getattr(exc, "_aichain_usage", None)
                raise
            self.last_usage = usage
            self.last_adaptations = made
            self.history    = [result]
            return result

        # Build the turn sequence: real messages interleaved with generate
        # points; add an implicit final generate when the script ends on an
        # unanswered ``user`` turn.
        seq: list = [("gen",) if adapters.is_generate_marker(m) else ("msg", m)
                     for m in messages]
        # Any turns after the last marker are a final question, whoever speaks
        # them: they were dropped, and the caller got the draft before them.
        if seq and seq[-1][0] == "msg":
            seq.append(("gen",))
        last_gen = max(k for k, s in enumerate(seq) if s[0] == "gen")

        # Published as the turns are paid for, so a call stopped between
        # turns — a budget, an error — leaves what was already spent readable.
        running: list = []
        history: list = []
        self.history, self.last_usage = history, None
        final: str | dict | None = None
        for k, item in enumerate(seq):
            if item[0] == "msg":
                running.append(item[1])
                continue
            output = self._output if k == last_gen else _TEXT_OUTPUT
            try:
                reply, usage, made = self._call_once(
                    model, running, output, max_retries, retry_delay)
            except Exception as exc:
                if (paid := getattr(exc, "_aichain_usage", None)) is not None:
                    self.last_usage = paid if self.last_usage is None else self.last_usage + paid
                raise
            self.last_adaptations = made
            self.last_usage = usage if self.last_usage is None else self.last_usage + usage
            history.append(reply)
            final = reply
            # Splice the reply into the context as an assistant turn so the next
            # turn sees it. A non-text (json) reply is stringified for context.
            text = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
            running.append({"role": "assistant",
                            "parts": [{"type": "text", "text": text}]})

        return final

    def _call_once(
        self,
        model:       "Model",
        messages:    list,
        output:      dict,
        max_retries: int,
        retry_delay: float,
    ) -> "tuple[str | dict, Usage, list]":
        """One model call with transient retries: ``(result, usage, adaptations)``.

        The adaptations come back with the request that made them; reading
        them off a Model shared between calls read another call's.
        """
        _tools = getattr(self, "_tools", None)
        own = isinstance(model, Model)
        if own:
            path, body, made = model.build(messages, output, _tools)
        else:
            # This boundary accepts anything model-shaped (tests stub bare
            # duck types), and a kwarg the double never heard of would break
            # every caller that wants no tools at all.
            path, body = (model.to_request(messages, output, tools=_tools) if _tools
                          else model.to_request(messages, output))
            made = list(getattr(model, "last_adaptations", ()) or ())

        for attempt in _attempts(max_retries, retry_delay):
            t0 = self._started(model)
            try:
                response = json.loads(model.client.send(path, body, model.client._auth_headers()))
                # ``cache_ttl`` is read leniently where ``name`` is not: a
                # Model always has it, but this boundary accepts anything
                # model-shaped, and the lifetime only picks a price multiplier.
                # A cost estimate is the last thing that should abort a call
                # that already succeeded.
                usage    = attach_cost(extract_usage(response), model.name,
                                       getattr(model, "cache_ttl", "5m"),
                getattr(model, "_provider", None))
                for budget in in_force(self.max_cost):
                    budget.charge(getattr(usage, "cost", None))
                try:
                    result = (model.from_response(response, output, tools_sent=bool(_tools)) if own
                              else model.from_response(response, output))
                except Exception as exc:
                    # Paid for and unparseable: the usage was lost with the
                    # exception, and no ended event said the call happened.
                    self._ended(model, t0, usage, error=exc)
                    exc._aichain_usage = usage
                    raise
                self._ended(model, t0, usage)
                return result, usage, made
            except APIError as exc:
                if hasattr(exc, "_aichain_usage"):
                    raise                         # a parse failure, already recorded
                self._ended(model, t0, error=exc)
                if not _retryable(exc, attempt, max_retries):
                    raise
                _pause(exc, attempt, retry_delay)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serialise this skill to a YAML file at *path*.

        The file stores everything needed to recreate the skill — the model
        name, input template, output spec, variables, name, and
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
        from .. import _yaml
        _yaml.write(path, self.to_dict(), "Skill.save()")

    def to_dict(self) -> dict:
        """What `save` writes, and a Chain's saved step holds.

        The model's own options, the fallback chain, the retries and a ceiling
        given as a number all come back: a saved Skill at temperature 0 came
        back at 1.0, with no fallback and no budget. A shared `Budget` cannot
        be written down — it is an object others hold — and saving one says so.
        """
        from .._budget import Budget
        if isinstance(self.max_cost, Budget) and self.max_cost is not self._own_budget:
            raise TypeError("this Skill holds a shared Budget, which a file cannot carry; "
                            "give it max_cost=<number> to save, or rebuild it in code")
        data = {
            "model_name":    self.model.name,
            "model_options": dict(getattr(self.model, "_asked", {}) or {})
                             | ({"extra": self.model.extra} if getattr(self.model, "extra", None) else {}),
            "fallback":      [m.name for m in self.models[1:]],
            "name":          self.name,
            "description":   self.description,
            "input":         self._input,
            "output":        self._output,
            "variables":     self.variables or None,
            "max_retries":   self.max_retries or None,
            "retry_delay":   self.retry_delay if self.retry_delay != 2.0 else None,
            "max_cost":      self.max_cost.limit if self.max_cost is not None else None,
        }
        return {k: v for k, v in data.items() if v not in (None, [], {})}

    @classmethod
    def from_dict(cls, data: dict, api_key: str | None = None) -> "Skill":
        """A Skill from what `to_dict` wrote.

        Parameters
        ----------
        data : dict
            What `to_dict` returned.
        api_key : str | None, optional
            The key for the model; its environment variable when omitted.
        """
        if "options" in data:
            raise ValueError("a saved Skill's 'options' were never applied and are not read; "
                             "a model's options go under 'model_options'")
        options = data.get("model_options") or None
        models = [Model(data["model_name"], api_key=api_key, options=options)]
        models += [Model(name, api_key=api_key) for name in data.get("fallback") or []]
        return cls(
            model       = models if len(models) > 1 else models[0],
            input       = data["input"],
            output      = data.get("output"),
            variables   = data.get("variables"),
            name        = data.get("name"),
            description = data.get("description"),
            max_retries = data.get("max_retries", 0),
            retry_delay = data.get("retry_delay", 2.0),
            max_cost    = data.get("max_cost"),
        )

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

            from yait_aichain.skills import Skill

            skill  = Skill.load("skills/translation.yaml")
            result = skill.run(variables={"text": "Hello, world!"})
        """
        from .. import _yaml
        return cls.from_dict(_yaml.read(path, "Skill.load()", ("model_name", "input")), api_key)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [f"model={getattr(self.model, 'name', self.model)!r}"]
        if self.name:
            parts.append(f"name={self.name!r}")
        if self.description:
            parts.append(f"description={self.description!r}")
        if self.variables:
            parts.append(f"variables={self.variables!r}")
        return f"Skill({', '.join(parts)})"
