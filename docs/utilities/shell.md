## `Shell` — a folder as a utility

```python
from yait_aichain.utilities import Shell

sh = Shell("projects/demo")
sh.read.run("README.md", {"start_line": 1, "end_line": 40})
Agent(model, tools=[sh.browse, sh.read])             # reads, and cannot write or run anything
Agent(model, tools=sh.tools)                         # all four
```

`browse`, `read`, `write` and `run` are tools, each confined to the folder and
each given to an agent on its own. `LocalTools(root_dir)` — all four at once —
still works.
