"""``Skill``: one model call made reusable — a Model, an input template with
``{placeholder}`` variables, a declared output format and default variables.
``run()`` substitutes, builds the request through the Model, sends it and
parses the reply; ``save()``/``load()`` keep it as YAML.
See docs/pipelines/skills.md.
"""


from .._budget import as_budget, in_force, start_run
import json
import random
import time
import warnings
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
from ..clients._families._structured import hit_ceiling, parse_structured
from ..models._usage import Usage, extract_usage, attach_cost
from .._events import Event, emit
from . import _adapters as adapters

#: Models already warned about a cut-off answer, so a loop says it once.
_CUT_SAID: set = set()

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


def _priced_as(model, made) -> str:
    """The model a call is priced as: the one it was swapped to, if it was."""
    for a in made or ():
        if getattr(a, "kind", None) == SWAPPED and getattr(a, "option", "") != "model" \
                and isinstance(getattr(a, "sent", None), str):
            return a.sent
    return model.name


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



def _input_of(input, prompt):
    """The messages spec: *input*, or one user message from *prompt*."""
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
        return {"messages": [{"role": "user", "parts": [prompt]}]}
    if input is None:
        raise ValueError(
            "Skill needs input= (the full messages spec) or prompt= (one "
            "user message).")
    return input


def _models_of(model) -> list:
    """One model or a fallback chain, as a list; the first is the primary."""
    # A single model or a fallback chain: on a transient failure
    # (rate limit / server / network) the next model is tried.  A bare
    # model is a chain of one — self.model is always the primary.
    models = list(model) if isinstance(model, (list, tuple)) else [model]
    if not models:
        raise ValueError("Skill requires at least one model.")
    named = [m for m in models if isinstance(m, str)]
    if named:
        raise TypeError(f"Skill takes a Model, not its name: Skill(Model({named[0]!r}), ...)")
    return models


class Skill:
    """
    A reusable, model-bound task unit.

    Parameters
    ----------
    model : Model | list[Model]
        The model that runs the skill; a list is a fallback chain.
    input : dict
        Universal message template, ``{"messages": [{"role", "parts"}]}``; text
        parts may hold ``{placeholder}`` tokens and may be plain strings. An
        ``assistant`` turn with no ``parts`` marks one model call, for
        several turns you direct.
    output : dict | None, optional
        Expected format; default ``{"modalities": ["text"], "format": {"type":
        "text"}}``, else e.g. ``{"format": {"type": "json"}}`` or
        ``"json_schema"`` with a ``schema``. Image output raises: use ``image()``.
    variables : dict | None, optional
        Default variable values; ``run(variables=...)`` overrides them.
    **refused
        ``options=`` is refused; a model's options are ``Model(name, options={…})``.
    name : str | None, optional
        Human-readable label for the skill.
    description : str | None, optional
        Short description of what the skill does.
    prompt : str | None, optional
        Shortcut for one user message; give ``prompt`` or ``input``, not both.
    max_retries : int, optional
        Retries after a transient failure (429, 5xx, network); ``0`` is one attempt.
    retry_delay : float, optional
        Seconds before the first retry; each later one waits twice as long.
    hooks : list | None, optional
        Callables given an ``Event`` for each model call.
    max_cost : float | Budget | None, optional
        A ceiling in dollars; a call that would begin past it raises
        ``BudgetExceeded``.

    Examples
    --------
    ::

        skill = Skill(Model("gpt-6-luna"), prompt="What is {topic}?")
        skill.run(variables={"topic": "gravity"})
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
        input = _input_of(input, prompt)
        input  = adapters.normalize_input(input)
        output = adapters.normalize_output(output)
        adapters.validate_input(input)
        adapters.validate_output(output)

        self.models = _models_of(model)
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
        #: Whether the last answer was cut at ``max_tokens``. Prose cut short
        #: is still returned — half an answer can be worth having — but as
        #: what it is: this says so, and a warning says it once per model.
        self.last_truncated: bool = False
        # Generated replies of the most recent run(), in order (multi-turn
        # directed reasoning); ``history[-1]`` is the returned value. A single-
        # shot run leaves a one-element list. ``None`` until the first call.
        self.history: list | None = None

        #: The whole answer from the last :meth:`stream`, assembled and
        #: parsed. None until a stream completes.
        self.last_result: str | dict | None = None

    def _emit(self, etype: str, **fields) -> None:
        """Dispatch an observability :class:`Event` to registered hooks."""
        emit(self.hooks, Event(type=etype, **fields))

    def _started(self, model) -> float:
        """Check the budget, announce the call, return its start time. Before the
        call: a reply's length is unknown until paid for, so a budget bounds
        beginning another call, never exceeding by a cent."""
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
            Variables for this call, merged over ``self.variables``.
        max_retries : int | None, optional
            Retries for this call only, over the constructor's; ``0`` is one attempt.
        retry_delay : float | None, optional
            First retry delay for this call only; it doubles on each retry.

        Returns
        -------
        str | dict
            Text, or a dict when the format is ``"json"`` / ``"json_schema"``.

        Raises
        ------
        APIError
            Non-transient, or transient (429, 5xx) after every retry and model.
        ValueError
            The response cannot be parsed; never retried.
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
        self.last_truncated = False
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

        Unlike ``run()``: retries stop at the first piece and there is no
        fallback chain; ``last_result`` holds the whole (parsed) answer at the
        end; ``last_usage`` is None when the provider reported nothing; a
        provider that cannot stream yields one piece and records a ``declined``.
        See ``docs/pipelines/skills.md``.

        Parameters
        ----------
        variables : dict | None, optional
            Variables for this call, merged over the Skill's own.
        """
        # Bound to the context it is made in: a stream made inside `with
        # using(ctx)` and read after it resolved its key with no tenant.
        from .._bound import bound
        return bound(self._stream(variables))

    def _stream(self, variables: "dict | None" = None):
        start_run(self)
        merged = {**self.variables, **(variables or {})}
        messages = adapters.substitute(self._input["messages"], merged)
        messages = adapters.resolve_media(messages)

        self.last_usage = None
        self.last_adaptations = []
        self.last_truncated = False
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
            # ends the generator with GeneratorExit, which would skip the
            # charge and the ended event.
            if not finished and self._open_call is not None:
                self._ended(model, self._open_call, error="abandoned")
            self._open_call = None

    _open_call: "float | None" = None

    def _stream_on(self, model, messages, _tools, pieces):
        try:
            result, usage, made = yield from self._call(
                model, messages, self._output, self.max_retries, self.retry_delay, pieces)
        except Exception as exc:
            self.last_usage = getattr(exc, "_aichain_usage", None)
            if pieces:
                self.last_result = "".join(pieces)     # the text, kept when it would not parse
            raise
        self.last_usage, self.last_adaptations = usage, made
        self.last_result = result
        self.history = [result]

    def _cut_short(self, model) -> None:
        """Mark the answer as cut at ``max_tokens``, and say so once per model."""
        self.last_truncated = True
        name = getattr(model, "name", str(model))
        if name in _CUT_SAID:
            return
        _CUT_SAID.add(name)
        warnings.warn(
            f"{name}: the answer was cut at max_tokens and is returned as far as it got "
            "(Skill.last_truncated is True); raise max_tokens for the whole of it.",
            stacklevel=4)

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
        """Run *messages* on one model: a call per generate marker (and after a
        trailing user turn), each reply appended for the next; only the last
        uses the real output format. ``last_usage`` sums, ``history`` keeps each."""
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
        # them, and get a reply of their own.
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

    def _call_once(self, model, messages, output, max_retries, retry_delay):
        """One buffered model call: ``(result, usage, adaptations)``."""
        calls = self._call(model, messages, output, max_retries, retry_delay)
        try:
            next(calls)
        except StopIteration as done:
            return done.value
        raise RuntimeError("a buffered call yielded a piece")   # pragma: no cover

    def _call(self, model, messages, output, max_retries, retry_delay, pieces=None):
        """One model call with retries, buffered or streamed into *pieces*:
        ``(result, usage, adaptations)``. A stream retries only while nothing
        has been yielded — after that a retry would replay what was seen."""
        _tools = getattr(self, "_tools", None)
        own = isinstance(model, Model)
        streamed = pieces is not None
        if streamed:
            made: list = []
        elif own:
            path, body, made = model.build(messages, output, _tools)
        else:
            # Anything model-shaped is accepted here (tests stub bare duck
            # types); a `tools=` the double never heard of would break it.
            path, body = (model.to_request(messages, output, tools=_tools) if _tools
                          else model.to_request(messages, output))
            made = list(getattr(model, "last_adaptations", ()) or ())

        for attempt in _attempts(max_retries, retry_delay):
            t0 = self._started(model)
            self._open_call = t0
            try:
                if streamed:
                    outcome = yield from _tee(model.stream(messages, output, tools=_tools), pieces)
                    if outcome is None:          # a model-shaped double that returns nothing
                        outcome = StreamOutcome(getattr(model, "last_stream_result", None),
                                                getattr(model, "last_stream_usage", None),
                                                list(getattr(model, "last_adaptations", []) or []))
                    raw, made = outcome.usage, list(outcome.adaptations)
                    cut = getattr(outcome, "truncated", False)
                else:
                    raw = json.loads(model.client.send(path, body, model.client._auth_headers()))
                    cut = hit_ceiling(raw)
                # Priced as the model that answered: a swap sends the request
                # to another model at another rate. A stream that reported no
                # usage has none, rather than a zero nobody measured.
                usage = attach_cost(extract_usage(raw), _priced_as(model, made),
                                    getattr(model, "cache_ttl", "5m"),
                                    getattr(model, "_provider", None)) if raw or not streamed else None
                for budget in in_force(self.max_cost):
                    budget.charge(getattr(usage, "cost", None))
                try:
                    result = (self._streamed_result(outcome, output, pieces, _tools) if streamed
                              else model.from_response(raw, output, tools_sent=bool(_tools)) if own
                              else model.from_response(raw, output))
                except Exception as exc:
                    # Paid for and unparseable: the usage rides the exception
                    # and an ended event says the call happened.
                    self._open_call = None
                    self._ended(model, t0, usage, error=exc)
                    exc._aichain_usage = usage
                    raise
                self._open_call = None
                self._ended(model, t0, usage)
                if cut and (not isinstance(result, ToolCallRequest) if streamed
                            else isinstance(result, str)):
                    self._cut_short(model)
                return result, usage, made
            except Exception as exc:
                if hasattr(exc, "_aichain_usage"):
                    raise                         # a parse failure, already recorded
                self._open_call = None
                self._ended(model, t0, error=exc)
                if isinstance(exc, APIError) and not pieces and _retryable(exc, attempt, max_retries):
                    _pause(exc, attempt, retry_delay)
                    continue
                raise

    def _streamed_result(self, outcome, output, pieces, _tools):
        """A stream's whole answer, parsed as ``run()`` parses it."""
        fmt = (output or {}).get("format") or {}
        kind = fmt.get("type", "text")
        whole = outcome.result
        if isinstance(whole, ToolCallRequest) and kind == "json_schema" and not _tools:
            # Anthropic's structured output is a forced call of a tool the
            # library made up; its arguments are the answer, as run() has it.
            whole = self._structured_call(whole, fmt)
        if whole is not None and not isinstance(whole, str):
            return whole
        if kind in ("json", "json_schema"):
            return parse_structured("".join(pieces),
                                    fmt.get("schema") if kind == "json_schema" else None,
                                    strip_fences=True)
        return "".join(pieces)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serialise this skill to a YAML file at *path*.

        Stores the model name, input, output, variables, name and description;
        API keys are **never** written (:meth:`load` reads them from the
        environment).

        Parameters
        ----------
        path : str
            Destination file path; parent directories are created.
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

        Parameters
        ----------
        path    : str
            Path to the YAML file.
        api_key : str | None, optional
            Key for the model; read from the provider's environment variable
            when omitted.

        Returns
        -------
        Skill
            A skill with a ready-to-use model.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ImportError
            If PyYAML is not installed.
        ValueError
            If no API key is available for the model's provider.
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
        # Names only: a repr goes into logs and tracebacks, and the values
        # are often a person's data.
        if self.variables:
            parts.append(f"variables={sorted(self.variables)}")
        return f"Skill({', '.join(parts)})"
