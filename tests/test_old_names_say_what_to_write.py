"""
A name from an older tutorial raises, and says what to write instead.

4.0 removed every compatibility path: nothing an earlier version spelled
differently keeps working. What remains is the error — an agent writing from
its weights reaches for `VectorDB` or `Embedding(...)`, and a bare
"cannot import name" costs it a guess where a named replacement costs a retry.
`llms.txt` promises this, so it is a test.
"""

import importlib
import re
import unittest

from yait_aichain import _hints


class TestOldNamesRaiseWithTheReplacement(unittest.TestCase):

    def test_every_listed_tool_name(self):
        for name, hint in _hints.TOOL_NAMES.items():
            with self.subTest(name=name):
                with self.assertRaises(ImportError) as caught:
                    exec(f"from yait_aichain.tools import {name}")
                self.assertIn(hint, str(caught.exception))

    def test_every_listed_old_path(self):
        for path, hint in _hints.MODULES.items():
            if not path.startswith("yait_aichain.tools."):
                continue
            with self.subTest(path=path):
                with self.assertRaises(ModuleNotFoundError) as caught:
                    importlib.import_module(path)
                self.assertIn(hint, str(caught.exception))

    def test_a_name_that_never_existed_is_a_plain_error(self):
        with self.assertRaisesRegex(ImportError, "cannot import name 'Nonsense'"):
            from yait_aichain.tools import Nonsense  # noqa: F401

    def test_what_the_documents_list_is_covered(self):
        for name in ("VectorDB", "vectorQuery", "Embedding", "Reranker", "LocalTools", "MCPTools",
                     "TTS", "STT", "convertToMD"):
            with self.subTest(name=name):
                self.assertIn(name, _hints.TOOL_NAMES)


class TestWhatWasRemovedIn46SaysWhatToWrite(unittest.TestCase):

    def test_agent_memory_and_its_backends(self):
        for name in ("AgentMemory", "MemoryBackend", "InMemoryBackend", "FileBackend"):
            with self.subTest(name=name), self.assertRaisesRegex(ImportError, "FileStore"):
                exec(f"from yait_aichain.agent import {name}")

    def test_the_methods_nothing_called(self):
        from yait_aichain.agent import Journal
        from yait_aichain.agent._result import AgentResult
        from yait_aichain.tools import embed
        for obj, name, hint in ((Journal(), "recent", "journal.entries"),
                                (Journal(), "progress_summary", "do_not_redo"),
                                (AgentResult(output="", success=True, stopped_by="answered",
                                             steps_taken=0, tokens_used=0, mode="agile"), "memory", "history"),
                                (embed("openai", model="text-embedding-3-small", api_key="k"),
                                 "batch", "embedder.run(texts)")):
            with self.subTest(name=name), self.assertRaisesRegex(AttributeError, re.escape(hint)):
                getattr(obj, name)


if __name__ == "__main__":
    unittest.main()
