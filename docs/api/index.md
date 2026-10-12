# API

Every public class, function and method, with its signature, every parameter,
what it returns and what it raises. Generated from the code's own signatures
and docstrings by `.github/scripts/docs.py`, and checked in CI: a page that differs
from the code fails the build, and a public name with no page fails a test.
What a thing is for, and how to use it well, is in the guides.

| Page | What is on it |
|---|---|
| [Model](model.md) | `Model`, `Usage`, adaptations, tool calls on the wire |
| [Skill](skill.md) | `Skill` — one model call |
| [Chain](chain.md) | `Chain`, `ChainResult` |
| [Pool](pool.md) | `Pool`, `PoolResult` |
| [Agent](agent.md) | `Agent`, `AgentResult`, stop conditions, beacons, the journal |
| [Tools — the base](tools/index.md) | `tool`, `Tool`, `FunctionTool`, `ToolResult`, `Record` |
| Tools by family | [search](tools/search.md) · [fetch](tools/fetch.md) · [decide](tools/decide.md) · [speech](tools/speech.md) · [image](tools/image.md) · [embed](tools/embed.md) · [rerank](tools/rerank.md) · [wait and gate](tools/wait-gate.md) |
| Utilities | [Vectors](utilities/vectors.md) · [Shell](utilities/shell.md) · [REST](utilities/rest.md) · [MCP](utilities/mcp.md) |
| [Control](control.md) | `Budget`, `PermissionPolicy`, approvals, `Event`, `Tracer` |
| [State](state.md) | `RunContext`, `Suspend`, stores, `RunDocument` |
| [Eval](eval.md) | `Eval`, `Case`, `Report`, the scorers |
| [Errors](errors.md) | every exception a call can raise |

Data rather than code — the model registry, what each provider does with each
option, environment variables, the YAML schema — is in [Reference](../reference/index.md).
