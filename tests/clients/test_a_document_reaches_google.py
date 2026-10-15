"""
A PDF is a PDF — by its name or its first bytes — and Gemini is sent it; a part
the wire cannot carry is left out and said, never dropped in silence.
"""

import os
import sys
import tempfile
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from yait_aichain import Model, Skill
from yait_aichain.testing import recorded
from yait_aichain.tools import media_part


def _file(name, data):
    path = os.path.join(tempfile.mkdtemp(), name)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def _sent(part):
    model = Model("gemini-3.5-flash-lite", api_key="k")
    with warnings.catch_warnings(), recorded(model) as rec:
        warnings.simplefilter("ignore")
        skill = Skill(model, input={"messages": [{"role": "user", "parts": ["Read it.", part]}]})
        skill.run()
    return rec.requests[0].body["contents"][0]["parts"], skill.last_adaptations


class TestADocumentReachesGoogle(unittest.TestCase):

    def test_a_pdf_is_known_by_its_name_and_by_its_bytes(self):
        self.assertEqual(media_part(_file("a.pdf", b"%PDF-1.4 x"))["source"]["mime"], "application/pdf")
        self.assertEqual(media_part(_file("a.bin", b"%PDF-1.4 x"))["source"]["mime"], "application/pdf")

    def test_gemini_is_sent_the_pdf(self):
        parts, made = _sent(media_part(_file("a.pdf", b"%PDF-1.4 x")))
        self.assertEqual(parts[1]["inlineData"]["mimeType"], "application/pdf")
        self.assertEqual(made, [])

    def test_a_document_it_cannot_read_is_declined_and_said(self):
        parts, made = _sent(media_part(_file("a.docx", b"PK\x03\x04 x")))
        self.assertEqual(len(parts), 1)
        self.assertIn(("declined", "document input"), [(a.kind, a.option) for a in made])


if __name__ == "__main__":
    unittest.main()
