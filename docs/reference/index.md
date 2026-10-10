# Reference

Data the library is built from, as tables generated from it. Classes and their
parameters are in the [API](../api/index.md).

| Page | What's in it |
|---|---|
| [Model registry](model-registry.md) | Every supported model by provider and task, plus query helpers. |
| [Parameters, per provider](parameters.md) | Three tables from one run: what you may ask for and with what values, what each provider declares it takes, and what actually happens on each model — passed, renamed, converted, dropped. Generated from real requests. |
| [Tools](tools.md) | Every built-in tool in one table: its name as a model sees it, its class, its key, its risk class — generated from the classes. |
| [Environment variables](environment-variables.md) | All required and optional env vars, grouped by provider and tool. |
| [YAML schema](yaml-schema.md) | The YAML format for persisting and loading Skills and Chains. |

For version history, see [CHANGELOG.md](../../CHANGELOG.md).

---

## Quick links

- **Which models exist?** → [Model registry](model-registry.md)
- **Does `top_k` reach this provider?** → [Parameters, per provider](parameters.md)
- **What env vars do I need?** → [Environment variables](environment-variables.md)
- **What does a saved Chain look like?** → [YAML schema](yaml-schema.md)
