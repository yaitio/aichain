# An agent as a chain step

Most real pipelines are mostly known and partly not. Fetch the sources,
**research the open question**, write the report, save it: three steps you can
list, and one whose path nobody can. Put the Agent where the path is unknown,
and keep everything around it a Chain.

```python
chain = Chain(steps=[
    (fetch("local"),  "sources",  {"url": "link"}),       # known: deterministic
    (research_agent,  "brief"),                           # unknown: the agent decides
    (write_report,    "report"),                          # known: one Skill
    (save_report,     "path"),                            # known: a function
], max_cost=1.00)

chain.run(variables={"link": "https://…", "task": "What changed in the market this quarter?"})
```

The agent gets the open-ended part and nothing else, so it is cheaper to run,
easier to read in a transcript and easier to replace — and the parts that
should be the same every run are.

→ The step's parameters: [API: Chain](../api/chain.md)

---

## What the agent sees

The chain calls the agent with its task — the `task` variable by default — and
hands every other variable along as context, so whatever earlier steps produced
is in front of the model without being pasted into the task by hand:

```python
(research_agent, "brief", {}, {"task_key": "research_question"})   # task from another variable
```

## What the chain gets back

The agent's answer, stored under the step's key, for the next steps to read as
`{brief}`. `output_field` picks another field of the `AgentResult` when you
need one. An agent that fails — a ceiling reached, a budget spent — fails the
step, and the chain's `on_step_error` decides what happens next, as with any
other step.

## One budget, both levels

`max_cost=` on the chain is shared with the agent inside it and the Skills
inside that: they spend from the same `Budget`. A copy at each level would let
every level spend the whole amount.

## Why not the other way round

An agent can call a chain as a tool, and sometimes should. But when the outer
shape is known, making an agent rediscover it costs a model turn per step and
gives a different path each run. The rule from [Chain](../pipelines/chain.md)
holds at every scale: write down what you can.

---

▶ [`examples/15_agent_orchestrator.py`](../../examples/15_agent_orchestrator.py)
