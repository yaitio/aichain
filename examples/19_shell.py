"""
19_shell.py — Shell under a policy that reads the command, not only the tool.

Keys: none. Runs offline: a scripted model asks for `git status`, then `git push`.
"""

import os
import subprocess
import tempfile
from pathlib import Path

from yait_aichain import Agent, Tracer
from yait_aichain.testing import ScriptedModel, call
from yait_aichain.utilities import Shell

OUT = Path(os.getenv("AICHAIN_EXAMPLES_DIR", tempfile.gettempdir())) / "aichain-examples"
repo = OUT / "repo"
repo.mkdir(parents=True, exist_ok=True)
subprocess.run(["git", "init", "-q", str(repo)], check=True)

shell = Shell(folder=str(repo))
run = next(t for t in shell.tools if t.name == "local_run")
commands = ["git status", "git push"]

# A real application passes Model("..."); the script plays the model here.
model = ScriptedModel([call("local_run", input=c) for c in commands] + ["Done."])
tracer = Tracer()
agent = Agent(model, tools=shell.tools, hooks=[tracer],
              policy={"rules": {"privileged": "deny"}})   # `read` is allowed by default
agent.run("Check the repository, then push it.")

ended = [e for e in tracer.events if e.type == "tool_call.ended"]
for command, event in zip(commands, ended):
    risk = run.risk_for({"input": command})
    print(f"{command}  {risk}  {'denied' if event.error else 'allowed'}")
print(f"\nrepository: {repo}")
