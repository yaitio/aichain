"""MarkItDown as an aichain tool. aichain does not convert documents; this is how
to bring a library that does. (cite's copy also reads scans; none are needed here.)"""

from pathlib import Path

from markitdown import MarkItDown

from yait_aichain.tools import tool

DATA = (Path(__file__).parent / "data").resolve()
_markitdown = MarkItDown()


@tool(risk="read")
def convert(path: str) -> dict:
    """Convert a local document — HTML, PDF, Word, PowerPoint, Excel or EPub — to
    text a model can read.

    Parameters
    ----------
    path : str
        A file under the recipe's data/ directory.
    """
    file = Path(path).resolve()
    if DATA not in file.parents:
        raise ValueError(f"{path}: convert reads files under {DATA} only")
    return {"source": path, "text": _markitdown.convert(str(file)).text_content or ""}
