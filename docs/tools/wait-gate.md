## Made by name

```python
from yait_aichain.tools import gate, wait
from yait_aichain.state import FileStore

Chain(steps=[(draft, "post"), (wait(reason="Approve the post?"), "approval"), (publish, "done")],
      store=FileStore("runs/"))
Agent(model, tools=[gate(refund, reason="Approve this refund?")], store=FileStore("runs/"))
```

`wait(...)` and `gate(tool, ...)` make `Wait` and `Gate`, with `description=`
beside the class's own settings. Until 3.5.0 they lived in `tools._wait`, which
still imports.
