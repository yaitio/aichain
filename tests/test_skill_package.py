"""
The Claude Code plugin is what the repository tests, generated and nothing else.
"""

import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals" / "agent_builds"))

_spec = importlib.util.spec_from_file_location("skill_script", ROOT / "scripts" / "skill.py")
skill = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(skill)

import yait_aichain  # noqa: E402


class TestThePlugin(unittest.TestCase):

    def test_every_generated_file_is_current(self):
        self.assertEqual(skill.main(["--check"]), 0, "run python scripts/skill.py")

    def test_the_plugin_carries_the_librarys_version(self):
        manifest = json.loads((skill.PLUGIN / ".claude-plugin" / "plugin.json").read_text())
        market = json.loads(skill.MARKETPLACE.read_text())
        self.assertEqual(manifest["version"], yait_aichain.__version__)
        self.assertEqual(market["plugins"][0]["version"], yait_aichain.__version__)
        self.assertEqual(market["plugins"][0]["name"], manifest["name"])
        self.assertTrue((ROOT / market["plugins"][0]["source"]).is_dir())

    def test_the_skill_has_a_name_and_a_description_that_fits(self):
        text = (skill.SKILL_DIR / "SKILL.md").read_text()
        front = text.split("---")[1]
        fields = dict(line.split(": ", 1) for line in front.strip().splitlines())
        self.assertEqual(fields["name"], "yait-aichain")
        self.assertLessEqual(len(fields["description"]), 1024)

    def test_the_files_the_skill_names_are_beside_it(self):
        text = (skill.SKILL_DIR / "SKILL.md").read_text()
        for name in ("llms.txt", "recipes.md"):
            with self.subTest(file=name):
                self.assertIn(f"`{name}`", text)
                self.assertTrue((skill.SKILL_DIR / name).exists())

    def test_no_held_out_task_reaches_the_plugin(self):
        from holdout import HOLDOUT
        from holdout2 import HOLDOUT2
        bundle = "".join(p.read_text() for p in skill.SKILL_DIR.iterdir())
        self.assertNotIn("holdout", bundle.lower())
        for t in HOLDOUT + HOLDOUT2:
            with self.subTest(task=t.id):
                self.assertNotIn(t.text, bundle)
                self.assertNotIn(t.reference.strip(), bundle)

    def test_every_tuning_reference_is_a_recipe(self):
        from tasks import TASKS
        recipes = (skill.SKILL_DIR / "recipes.md").read_text()
        for t in TASKS:
            with self.subTest(task=t.id):
                self.assertIn(t.reference, recipes)


if __name__ == "__main__":
    unittest.main()
