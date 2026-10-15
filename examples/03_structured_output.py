"""
03_structured_output.py — Data from a model, not prose: an answer that fits a JSON schema.

Keys: OPENAI_API_KEY.
"""

import json

from yait_aichain import Model, Skill, check_structure

EMAIL = ("Hi, this is Dana Reyes. My order 88-1041 arrived yesterday, "
         "but the blender inside was cracked. Can you send a replacement?")
SCHEMA = {"type": "object",
          "properties": {"name": {"type": "string"}, "order_id": {"type": "string"},
                         "problem": {"type": "string"}},
          "required": ["name", "order_id", "problem"]}

# Key: OPENAI_API_KEY from the environment; or Model("gpt-5.4-mini", api_key="...")
skill = Skill(Model("gpt-5.4-mini"), prompt="Extract the details from this email: {email}",
              output={"format": {"type": "json_schema", "schema": SCHEMA}})
result = skill.run(variables={"email": EMAIL})
print(json.dumps(result, indent=2))
print(f"problems: {check_structure(result, SCHEMA)}")
