"""
`AgentMemory` and `FileBackend` still work, and say they are going.

Nothing in the library reads them — an agent's state is its conversation — so
they leave in 5.0. Until then a caller who built on them is told, not broken.
"""

import os
import tempfile
import unittest

from yait_aichain.agent import AgentMemory, FileBackend


class TestDeprecated(unittest.TestCase):

    def test_agent_memory_warns_and_works(self):
        with self.assertWarns(DeprecationWarning) as caught:
            memory = AgentMemory({"a": 1})
        self.assertIn("5.0", str(caught.warning))
        memory.set("b", 2)
        self.assertEqual((memory.get("a"), memory.get("b")), (1, 2))

    def test_file_backend_warns_and_persists(self):
        path = os.path.join(tempfile.mkdtemp(), "m.json")
        with self.assertWarns(DeprecationWarning):
            memory = AgentMemory(backend=FileBackend(path))
        memory.set("k", "v")
        memory.flush()
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(AgentMemory(backend=FileBackend(path)).get("k"), "v")


if __name__ == "__main__":
    unittest.main()
