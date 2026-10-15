# Tools

A model reads and writes text. A tool does the rest: searches the web, reads a
page, counts, draws, calls an API, waits for a person. A tool is anything your
code, a chain and an agent can all call, the same way:

```python
from yait_aichain.tools import search

found = search("exa").run("managed vector databases", {"max_results": 5})
found["results"][0]["url"]       # fields, for code
str(found)                       # one readable text, which is what a model is shown
```

---

## What a tool is

Three things, and the model sees two of them:

- a **name** and a **description** — what a model reads to decide whether to
  call it;
- a **schema** — the arguments it takes, as JSON Schema;
- **`run()`** — the code that does the work.

What a tool *is* — the service, the model, the folder, the key — is set when
it is made. What one call asks goes in `.run()`. A setting never appears in
both places:

```python
search("exa")                                    # made: which service
    .run("vector databases", {"domains": ["arxiv.org"], "max_results": 5})   # asked: this call
```

`run()` returns the data, and raises on failure. Calling the tool itself —
`tool(...)` — returns a `ToolResult` that never raises: `.success`, `.output`,
`.error`, and `bool(result)` is `.success`. Chains and agents use that form.

---

## Kinds of tools

### Built-in tools, made by service name

```python
from yait_aichain.tools import search, fetch, chunk, embed, rerank, image

search("perplexity")                             # web search: perplexity, brave, exa, firecrawl, tavily, parallel
fetch("local")                                   # a page by URL: firecrawl, tavily, exa, parallel, local
chunk(max_chars=800)                             # text → pieces
embed("openai", model="text-embedding-3-small")  # texts → vectors
rerank("cohere", model="rerank-v3.5")            # documents → in order of relevance
image("openai", model="gpt-image-1.5", output_dir="images")   # prompt → picture
```

Each reads its key from the environment, and its name and description come
from the service — `search("exa")` is `searchExa` to a model. `name=` and
`description=` override both.

**When to use:** whenever one exists. The list, with keys, risks and every
parameter: [Tools reference](../reference/tools.md).

### Your own function, with `@tool`

```python
from yait_aichain.tools import tool

@tool(risk="read")
def word_count(text: str, unique: bool = False) -> dict:
    """Count the words in a text.

    text: the text to count.
    unique: count each word once.
    """
    words = text.split()
    return {"words": len(set(words)) if unique else len(words)}

word_count.run(text="to be or not to be")       # {'words': 6}
```

The function's name is the tool's name, the first line of its docstring the
description, its annotated parameters the schema, each described by its
`name: text` line. A function with no docstring, or a parameter with no JSON
type, is refused when the tool is made. `risk` defaults to `"write"`: the
library cannot tell what a function does, so saying `"read"` is yours to do.

**When to use:** almost always, for your own code. A chain holding a `@tool`
cannot be saved to YAML — a function is code.

### A `Tool` subclass

```python
from yait_aichain.tools import Tool

class Counter(Tool):
    name        = "counter"
    description = "Add to a running total and return it."
    parameters  = {"type": "object",
                   "properties": {"input": {"type": "number"}},
                   "required": ["input"]}

    def __init__(self):
        self.total = 0

    def run(self, input, options=None):
        self.total += input
        return self.total
```

**When to use:** the tool holds state between calls, or needs a schema
annotations cannot say.

### A utility's tool

```python
from yait_aichain.utilities import REST, Shell

api = REST("https://api.example.com", description="The shop's order API.")
get_order = api.endpoint("get_order", "GET", "/orders/{id}", path_params=["id"],
                         description="Return one order by its id.")
sh = Shell("notes")

Agent(model, tools=[get_order, sh.browse, sh.read])
```

A utility — `Vectors`, `Shell`, `REST`, `MCP` — is an object your code holds;
some of its members are tools, and those are what an agent gets.

**When to use:** an outside API, an MCP server, a folder, a vector store.
→ [Utilities](../utilities/index.md)

### Comparison

| | Who writes the schema | Settings | Saved with a chain | Typical use |
|---|---|---|---|---|
| Built-in | the library | the factory: service, model, key | yes | search, fetch, chunk, embed, rerank, images, speech |
| `@tool` | your annotations and docstring | `name`, `description`, `risk` | no | your own code |
| `Tool` subclass | you, by hand | your `__init__` | yes, if its arguments are | a tool with state |
| A utility's tool | the utility | the utility: URL, folder, server, collection | the utility decides | APIs, MCP servers, folders, vector stores |

---

## In a chain

A tool step gets the arguments its schema names, from the chain's variables.
The step's third element maps them — to a variable's name, or to a string with
`{variables}` the chain fills in:

```python
chain = Chain(steps=[
    (fetch("local"),  "page",    {"url": "link"}),                 # the variable `link`
    (word_count,      "count",   {"text": "Title: {title}"}),      # a template
    (summarise,       "summary"),
])
chain.run(variables={"link": "https://example.com", "title": "the quick brown fox"})
```

A template naming a variable the chain does not have is an error with its name.
What a tool returns is kept under its key; a plain dict is merged into the
variables, so one step can write several. → [Chain](../pipelines/chain.md)

## In an agent

```python
agent = Agent(Model("gpt-6-luna"), tools=[search("exa"), fetch("local"), word_count])
```

The agent offers each tool's name, description and schema to the model, and
checks every call against the schema before it runs: a missing argument or an
unknown key comes back to the model as a message it can correct, not as a
crash. Two tools with one name, or something that is not a tool, are refused
when the agent is made. → [Agent](../agents/overview.md)

---

## Risk and permission

`risk` is a property of the tool — `read`, `draft`, `write` (the default),
`external`, `financial`, `destructive`, `privileged` — and an agent's
`policy` reads it before the tool runs: `allow` runs it, `deny` never does,
`approve` asks the policy's `approve` and runs only on a yes.

```python
from yait_aichain.tools import ApprovalDecision

agent = Agent(model, tools=[issue_refund],
              policy={"rules": {"financial": "approve"},
                      "approve": lambda request: request.arguments["amount"] <= 50
                                 or ApprovalDecision(False, "over $50 needs a ticket")})
```

With an `approve` rule and no one to answer it, the call is refused, and the
refusal reaches the model with its reason. → [Permissions](../control/permissions.md)

---

## Keys and untrusted input

A tool's `api_key=` is a string, or a callable asked on each request with the
run's `RunContext` — the same rule as `Model(api_key=callable)`, so one tool
serves every tenant. A resolver that returns nothing raises rather than
falling back to the process-wide key. → [Production](../production.md)

A URL or an output path a model supplies is untrusted. The built-in tools
fetch only `http`/`https`, refuse hosts that resolve to a private or internal
address, and never follow a redirect to another host with the caller's
headers. For a trusted deployment, `AICHAIN_ALLOW_PRIVATE_URLS=1` allows
private hosts, and `AICHAIN_OUTPUT_ROOT=/path` confines every file a tool
writes to that folder.

---

## Wait and gate

Two tools that pause instead of returning: `wait(reason=…)` parks a run until a
signal arrives, and `gate(tool)` runs a tool only when the signal approves. In
a chain or in an agent's hands, with a `store=`, the run resumes later in
another process. → [Pause and resume](../pipelines/pause-and-resume.md)

---

▶ [`examples/05_tool_fetch.py`](../../examples/05_tool_fetch.py) ·
[`examples/07_tool_custom.py`](../../examples/07_tool_custom.py) ·
[`examples/14_agent_tools.py`](../../examples/14_agent_tools.py)
