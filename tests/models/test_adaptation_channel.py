"""
The adaptation channel tells the truth under `-W error` and does not read the
conversation as options.

The mark was set before the warning, so under `filterwarnings=error` the
first call raised with an empty record and every later one passed in
silence; `note_absent` searched the whole body, so a "1h" or a 4096 in a
tool's arguments made a declined option look delivered.
"""

import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain.models import Model
from yait_aichain.models._adaptation import reset_warnings


class TestChannel(unittest.TestCase):

    def setUp(self):
        reset_warnings()

    def test_under_error_every_call_raises_and_leaves_its_record(self):
        m = Model("gpt-5.5", api_key="k", options={"temperature": 0.3})
        msg = [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for _ in range(2):
                with self.assertRaises(RuntimeWarning):
                    m.build(msg, {"format": {"type": "text"}})
                self.assertTrue(m.last_adaptations)

    def test_a_value_in_the_conversation_is_not_a_delivered_option(self):
        m = Model("deepseek-chat", api_key="k", options={"cache_ttl": "1h"})
        msg = [{"role": "user", "parts": [{"type": "text", "text": "hi"}]},
               {"role": "assistant", "tool_calls": [{"id": "1", "name": "t", "arguments": {"window": "1h"}}]},
               {"role": "tool", "call_id": "1", "parts": [{"type": "text", "text": "1h"}]}]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, made = m.build(msg, {"format": {"type": "text"}})
        self.assertIn("cache_ttl", [a.option for a in made])


if __name__ == "__main__":
    unittest.main()
