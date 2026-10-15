# Chain

A Chain is a pipeline whose steps you can list: summarise, then translate,
then count the words, then publish. The steps run in order and share one set
of variables, so a step reads what the earlier ones wrote without any code to
pass it along.

```python
chain = Chain(steps=[
    (summarise, "summary"),
    (translate, "french"),
])
result = chain.run(variables={"text": "…", "language": "French"})
result["french"], result.output, result.cost
```

→ Every parameter: [API: Chain](../api/chain.md)

---

## When a Chain, and when an Agent

When you can write the steps down, write them down. A Chain does exactly what
it lists: no turn spent deciding what to do next, the same path every run, and
a cost you can predict from the steps. An Agent earns its price only where the
next step depends on what the last one returned. The two compose: an Agent can
be one step of a Chain, and the Chain hands it its task.
→ [An agent as a chain step](../agents/agent-as-chain-step.md)

## Variables are the spine

A run starts with the Chain's default variables, overridden by the ones given
to `run()`. Each step adds its output under its key, so by the last step the
variables hold everything the run computed. That is also what the result
reads: `result["summary"]` is any step's output by its key, `result.output` the
last step's.

Each kind of step reads them differently, because each needs something
different:

| Step | Reads | Writes |
|---|---|---|
| a `Skill` | every variable, as `{placeholders}` in its prompt | its answer under its key |
| a tool | only the arguments its schema names | its result under its key |
| a function, or a utility's method | the parameters its signature names | what it returns under its key |
| an `Agent` | its task from the `task` variable, everything else as context | its answer under its key |

A plain `dict` a step returns is merged into the variables instead — a REST
reply's fields become variables of their own. A tool's `Record` is not merged:
it is one result, so two search steps do not overwrite each other's
`results`, and a Skill reading `{found}` gets its text.

## Filling a tool's arguments

A tool takes its arguments by name, and the step's third element says where
each comes from:

```python
(fetch("local"),  "page",  {"url": "link"})                        # the variable `link`
(search("exa"),   "found", {"input": "news about {topic} this week"})   # a template
(rerank_tool,     "top",   {"input": "{candidates}",               # the value itself, a list stays a list
                            "options": {"query": "{question}", "top_n": 5}})
```

A template assembles the input by substitution, which is free and gives the
same text every time. When the input needs judgement — a good search query, an
image prompt — put a Skill before the tool and let a model write it: one call
more, a better input. Choosing between the two is choosing where you want the
model's judgement and where you do not.

A name or a template naming a variable the chain does not have is an error
with its name, and so is an argument the tool does not take: a pipeline that
silently passes nothing fails later, somewhere less obvious.

## A step in full

A tuple says the common cases: a runner alone, stored under the runner's name;
`(runner, key)`; `(runner, input_map)`; `(runner, key, input_map)`; and, for an
Agent, `(runner, key, input_map, options)`. `Step` says the rest:

```python
Step(count_words, out="words", inputs={"text": "page.body"}, on_error="skip", spread=False)
```

An input can be a path into a variable: `"page.body"` is the `body` field of
the dict stored under `page`. Where a step's fields are known before it runs —
a Skill answering to a JSON schema, a function returning a `TypedDict` or a
dataclass — a path to a field it does not have is refused when the chain is
built, with the closest one named; elsewhere it is found when the run reaches
it. A pause's answer (`wait`, `gate`) is spread among the variables, which is
what it is for. `on_error` is this step's own policy, over the
chain's. `spread` says whether a dict output's fields are also put among the
variables: `None`, the default for now, still spreads them, with a warning;
`True` spreads them on purpose; `False` keeps the dict under its key alone.

After a run, `chain.history` holds one record per step — `step`, `kind`,
`name`, `input`, `output`, `output_key`, `options`, and for a failed step
`error` and `failure`. Secrets are left out of `input`.

## Plain code is a step

A function is a step as it is; its parameters are filled from the variables by
name:

```python
def shout(slogan: str) -> str:
    return slogan.upper()

Chain(steps=[(write_slogan, "slogan"), (shout, "loud")])
```

So is a utility's method — `kb.upsert` writing what a chunking step produced.
Not every step needs a model, and a pipeline that wraps plain code in a prompt
to get it into a chain pays for every run of it.

## When a step fails

<!-- g:policy -->
| Mode | Behaviour |
|---|---|
| `"raise"` (default) | Propagate the exception. The run is marked failed and nothing after the step runs. |
| `"stop"` | End the run without raising; the error is in `history` and `run()` returns the last successful output (or `None`). |
| `"skip"` | Record the error, emit a `RuntimeWarning`, and run the next step. A later step that reads the failed step's output gets a stale or absent value. |
| `"collect"` | As `skip`, without the warning — for a long generated chain that expects a few steps to fail. |
<!-- /g:policy -->

The same four words mean the same thing on a `Pool`. `raise` is the default
because a failure you did not plan for should be loud; `skip` warns because a
later step may read a value that is now stale.

## Money, pauses, saving

- **`max_cost=`** is one ceiling for the whole chain, shared by every step:
  nobody knows in advance which step will be the expensive one, so it is not
  divided between them. → [Cost and budget](../control/cost-and-budget.md)
- **A step can pause the run** — `wait()` for a signal, `gate(tool)` for an
  approval — and `resume()` carries on later, in another process, from the
  first step not yet done. → [Pause and resume](pause-and-resume.md)
- **`chain.save()`** writes the steps to YAML, keys never included. A chain
  holding a `@tool` function or a plain function cannot be saved — a function
  is code, not data — and `save()` says so instead of writing a file that
  cannot be loaded. A REST endpoint is saved with its URL, fields and risk
  class; its credential only by name (`auth={"env": …}`), and a token given
  as a value refuses to be saved. `Chain.load()` imports only the library's
  own classes, and an endpoint that carries auth, or a risk class other than
  its method's, loads only with `trusted=True`: the URL and the variable a
  token is read from both come from the file.

---

▶ [`examples/20_chain.py`](../../examples/20_chain.py) ·
[`examples/21_chain_steps.py`](../../examples/21_chain_steps.py) ·
[`examples/22_chain_save_load.py`](../../examples/22_chain_save_load.py)
