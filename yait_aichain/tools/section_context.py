"""
tools.section_context
=====================

``SectionContextTool`` — rolling-context queue manager for sectional reports.

Stores compact summaries of written sections and makes them available
to subsequent section writers, ensuring narrative coherence without
passing full section text through the context window.
"""

from ._base import Tool


class SectionContextTool(Tool):
    """
    Rolling-context queue for sectional report writing.

    Each time a section is written, its summary is stored here.
    The next section writer retrieves all previous summaries to
    understand what has already been established and avoid repetition.

    Parameters
    ----------
    max_sections : int
        Maximum number of section summaries to keep in the queue.
        Oldest entries are dropped when the limit is exceeded.
    """

    name        = "section_context"
    description = (
        "Store or retrieve summaries of written report sections "
        "to maintain narrative coherence across the document."
    )
    parameters  = {
        "type": "object",
        "properties": {
            "action": {
                "type":        "string",
                "enum":        ["store", "retrieve"],
                "description": "store: save a section summary. retrieve: get all previous summaries.",
            },
            "section_title": {
                "type":        "string",
                "description": "Title of the section. Required for store.",
            },
            "summary": {
                "type":        "string",
                "description": "Compact summary of what the section covers. Required for store.",
            },
        },
        "required": ["action"],
    }

    def __init__(self, max_sections: int = 5) -> None:
        self._sections: list[dict] = []
        self._max = max_sections

    def run(
        self,
        action: str,
        section_title: str = "",
        summary: str = "",
        options=None,
    ) -> str:
        if action == "store":
            self._sections.append({"title": section_title, "summary": summary})
            if len(self._sections) > self._max:
                self._sections.pop(0)
            return f"Stored context for: {section_title}"

        if action == "retrieve":
            if not self._sections:
                return "No previous sections written yet."
            lines = ["Previously written sections:"]
            for s in self._sections:
                lines.append(f"- {s['title']}: {s['summary']}")
            return "\n".join(lines)

        return f"Unknown action: {action}"
