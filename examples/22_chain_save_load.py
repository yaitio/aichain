"""
22_chain_save_load.py — A pipeline as a file, with its policy: stored, versioned, run elsewhere.

Keys: OPENAI_API_KEY, ANTHROPIC_API_KEY. Writes writer_reviewer.yaml to AICHAIN_EXAMPLES_DIR.
"""

import os
import tempfile
from pathlib import Path

from yait_aichain import Chain, Model, Skill

OUT = Path(os.getenv("AICHAIN_EXAMPLES_DIR", tempfile.gettempdir())) / "aichain-examples"
OUT.mkdir(parents=True, exist_ok=True)
path = OUT / "writer_reviewer.yaml"

# Keys: OPENAI_API_KEY and ANTHROPIC_API_KEY from the environment; or Model(name, api_key="...")
chain = Chain([
    Skill(Model("gpt-5.4-mini"), name="writer", prompt="Write three sentences about {topic}."),
    Skill(Model("claude-haiku-5-5"), name="reviewer", prompt="Suggest one improvement:\n\n{writer}"),
], name="writer_reviewer", policy={"rules": {"external": "deny"}})
chain.save(str(path))
print(path.read_text())

loaded = Chain.load(str(path))                 # the key is read from the environment again
print(f"[loaded]\n{loaded.run(variables={'topic': 'tidal power'})['reviewer']}")
