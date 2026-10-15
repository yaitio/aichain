"""
08_skill_save_load.py — A prompt kept in a file apart from the code; the key never reaches it.

Keys: ANTHROPIC_API_KEY. Writes validator.yaml to AICHAIN_EXAMPLES_DIR (the temp directory by default).
"""

import os
import tempfile
from pathlib import Path

from yait_aichain import Model, Skill

OUT = Path(os.getenv("AICHAIN_EXAMPLES_DIR", tempfile.gettempdir())) / "aichain-examples"
OUT.mkdir(parents=True, exist_ok=True)
path = OUT / "validator.yaml"

# Key: ANTHROPIC_API_KEY from the environment; or Model("claude-haiku-5-5", api_key="...")
Skill(Model("claude-haiku-5-5"), name="validator",
      prompt="From 0 to 10, how likely is it that a person wrote this text? "
             "Answer with the number and one reason.\n\n{text}").save(str(path))
print(path.read_text())

skill = Skill.load(str(path))
print("answer:", skill.run(variables={
    "text": "Our quarterly numbers were fine. Honestly, the coffee machine broke and morale tanked."}))
