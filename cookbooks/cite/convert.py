"""MarkItDown as an aichain tool. aichain does not convert documents; this is how
to bring a library that does, and to read a page with no text through a vision model."""

import os
from pathlib import Path

from markitdown import MarkItDown

from yait_aichain import Model, Skill
from yait_aichain.tools import media_part, tool

DATA = (Path(__file__).parent / "data").resolve()
#: Not Claude: every Claude model answered a whole-page transcription with
#: "Output blocked by content filtering policy" (2026-10-17).
VISION_MODEL = os.getenv("VISION_MODEL", "gemini-3.8-flash")
IMAGES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
_markitdown = MarkItDown()


def _transcribe(path: Path) -> tuple[str, dict]:
    skill = Skill(Model(VISION_MODEL), input={"messages": [{"role": "user", "parts": [
        media_part(str(path)), "Transcribe this page exactly. Return only its text."]}]})
    text = skill.run()
    u = skill.last_usage
    return text, {"input_tokens": u.input_tokens + u.image_input_tokens,
                  "output_tokens": u.output_tokens, "cost": u.cost}


@tool(risk="read")
def convert(path: str) -> dict:
    """Convert a local document — PDF, Word, PowerPoint, Excel, HTML, EPub or an
    image — to text a model can read.

    Parameters
    ----------
    path : str
        A file under the recipe's data/ directory.
    """
    file = Path(path).resolve()
    if DATA not in file.parents:
        raise ValueError(f"{path}: convert reads files under {DATA} only")
    text = _markitdown.convert(str(file)).text_content or ""
    pages = text.split("\f") if "\f" in text else [text]
    scanned, usage = [], None
    for i, page in enumerate(pages):
        if len("".join(page.split())) < 20 and file.suffix.lower() in IMAGES:
            pages[i], usage = _transcribe(file)
            scanned.append(i)
    result = {"source": path, "text": "\n".join(pages), "pages": pages, "scanned": scanned}
    if usage:
        result.update(usage=usage, cost=usage["cost"])
    return result
