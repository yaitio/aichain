"""
tools.section_context — SectionContextTool
===========================================

A zero-cost, zero-API-call Tool that advances a section pointer in the
Chain's accumulated variable dict and maintains a rolling context window.

Role in the sectional document pipeline
----------------------------------------
When a Chain is built dynamically from a section list, all sections are
serialised into an initial variable called ``section_queue`` — a JSON-
encoded list of section metadata objects.  Before each section's generation
steps run, a ``SectionContextTool`` step pops the front item from the queue
and writes the current section's metadata into well-known variable names that
the downstream Skill / Agent prompts can reference via ``{current_*}``
placeholders.

Rolling context window
----------------------
In addition to advancing the queue, the tool rotates the rolling context
buffer so that write skills always have access to the last two section
summaries.  The rotation works as follows on each call:

  two_sections_ago_summary  ← old ``prev_section_summary``
  prev_section_summary      ← old ``current_section_actual_summary``
  recent_summaries          ← formatted block for use in write-skill prompts
  current_section_actual_summary ← reset to "" for the new section

Variables written on each call
--------------------------------
``current_section_id``              — machine identifier
``current_section_title``           — human-readable title
``current_section_plan``            — planned summary for this section
``current_section_position``        — integer position in final document
``current_section_runner``          — ``"skill"`` or ``"agent"``
``current_section_sources``         — reset to ``""`` for the new section
``current_section_actual_summary``  — reset to ``""`` for the new section
``section_queue``                   — updated queue (consumed item removed)
``prev_section_summary``            — summary of the section just written
``two_sections_ago_summary``        — summary from two sections ago
``recent_summaries``                — formatted rolling context block (ready
                                       for ``{recent_summaries}`` in prompts)

The tool raises ``RuntimeError`` when the queue is empty so an accidental
extra call is caught immediately rather than silently overwriting context with
stale values.

Usage
-----
    from tools import SectionContextTool

    tool   = SectionContextTool()
    result = tool(
        section_queue      = json_queue_string,
        recent_summaries   = "",      # optional — passed through unchanged
    )
    # result.output is a dict; Chain merges it into accumulated automatically

In a Chain step list::

    (SectionContextTool(), "section_ctx", {})

Because ``run()`` returns a ``dict``, Chain calls
``accumulated.update(output)`` — all variables above land directly
in the accumulated dict without any ``output_key`` collision.

The ``input_map`` is not needed: the tool reads all its inputs by their
exact names from the accumulated dict (passed as kwargs by Chain's
``_build_tool_kwargs``).
"""

from __future__ import annotations

import json

from ._base import Tool


def _format_rolling_context(two_ago: str, prev: str) -> str:
    """
    Assemble a formatted rolling context block from the last two summaries.

    Returns an empty string when both summaries are empty.
    """
    two_ago = (two_ago or "").strip()
    prev    = (prev    or "").strip()

    if not two_ago and not prev:
        return ""

    parts: list[str] = []
    header = "ROLLING CONTEXT (recently written sections)"

    if two_ago:
        parts.append(f"Two sections ago:\n{two_ago}")
    if prev:
        parts.append(f"Previous section:\n{prev}")

    return header + "\n" + "=" * len(header) + "\n" + "\n\n".join(parts)


class SectionContextTool(Tool):
    """
    Advance the section queue, rotate the rolling context, and expose the
    next section's metadata.

    This tool has no external dependencies and makes no API calls.
    It is purely a state-management primitive for the sectional document
    generation pipeline.

    Parameters
    ----------
    None — stateless; all state lives in the accumulated variable dict.

    Examples
    --------
    Standalone use::

        import json
        from tools import SectionContextTool

        sections = [
            {"id": "situation",   "title": "Situation Analysis",
             "plan": "Current market and competitive context.",
             "position": 1, "runner": "skill"},
            {"id": "competitive", "title": "Competitive Analysis",
             "plan": "Key players, positioning, white spots.",
             "position": 2, "runner": "agent"},
        ]

        tool   = SectionContextTool()
        result = tool(section_queue=json.dumps(sections))

        if result:
            print(result.output["current_section_title"])   # "Situation Analysis"
            # Remaining queue has 1 item
            remaining = json.loads(result.output["section_queue"])
            print(len(remaining))   # 1
    """

    name        = "section_context"
    description = (
        "Advance the section queue by one position, rotate the rolling "
        "context window, and write the next section's metadata (id, title, "
        "plan, position, runner) into the accumulated variable dict.  "
        "Returns a dict that Chain merges automatically.  "
        "Zero API calls — pure Python state management."
    )
    parameters  = {
        "type": "object",
        "properties": {
            "section_queue": {
                "type":        "string",
                "description": (
                    "JSON-encoded list of section metadata objects.  "
                    "Each object must have 'id', 'title', 'plan', "
                    "'position', and 'runner' keys."
                ),
            },
            "current_section_actual_summary": {
                "type":        "string",
                "description": (
                    "Summary (3–5 bullets) produced by the summarise step "
                    "for the section that was just written.  "
                    "Empty string on the very first call."
                ),
            },
            "prev_section_summary": {
                "type":        "string",
                "description": (
                    "The summary that was 'previous section' in the prior "
                    "call — rotated into 'two sections ago' by this call."
                ),
            },
            "two_sections_ago_summary": {
                "type":        "string",
                "description": (
                    "The summary that was 'two sections ago' in the prior "
                    "call — dropped by this call (only last 2 are kept)."
                ),
            },
        },
        "required": ["section_queue"],
    }

    # ------------------------------------------------------------------
    # Override __call__ to accept **kwargs (framework-internal tool with
    # named parameters rather than the standard input/options interface).
    # ------------------------------------------------------------------

    def __call__(self, **kwargs) -> "ToolResult":  # type: ignore[override]
        """Safe wrapper — accepts named kwargs, never raises."""
        from ._base import ToolResult
        try:
            output = self.run(**kwargs)
            return ToolResult(success=True, output=output)
        except Exception as exc:
            return ToolResult(success=False, output=None, error=str(exc))

    # ------------------------------------------------------------------

    def run(  # type: ignore[override]
        self,
        section_queue:                  str,
        current_section_actual_summary: str = "",
        prev_section_summary:           str = "",
        two_sections_ago_summary:       str = "",   # accepted but not used beyond rotation
    ) -> dict:
        """
        Pop the first section from *section_queue*, rotate the rolling
        context window, and return updated metadata as a flat dict.

        Parameters
        ----------
        section_queue : str
            JSON-encoded list of section metadata dicts.
        current_section_actual_summary : str, optional
            Summary of the section just completed (output from the
            summarise Skill step).  Empty on the first call.
        prev_section_summary : str, optional
            Value of ``prev_section_summary`` from the previous call.
            Used for the rotation: prev → two_sections_ago.
        two_sections_ago_summary : str, optional
            Accepted for parameter consistency but not carried forward
            (only the last two summaries are kept).

        Returns
        -------
        dict
            Contains all ``current_*`` variables, updated rolling context
            variables, and the trimmed ``section_queue`` string.
            Chain merges this dict into the accumulated variable dict.

        Raises
        ------
        RuntimeError
            When *section_queue* is empty.
        ValueError
            When *section_queue* is not valid JSON or a required key is
            missing.
        """
        try:
            queue: list = json.loads(section_queue)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"SectionContextTool: section_queue is not valid JSON: {exc}"
            )

        if not queue:
            raise RuntimeError(
                "SectionContextTool: section_queue is empty — no more "
                "sections to process.  This call was unexpected."
            )

        section = queue.pop(0)

        # Validate required keys
        for key in ("id", "title", "plan", "position", "runner"):
            if key not in section:
                raise ValueError(
                    f"SectionContextTool: section object is missing "
                    f"required key {key!r}.  Got: {list(section.keys())}"
                )

        # ── Rotate rolling context window ──────────────────────────────
        # After this call:
        #   two_sections_ago  ← old prev_section_summary
        #   prev_section      ← old current_section_actual_summary
        new_two_sections_ago = (prev_section_summary           or "").strip()
        new_prev             = (current_section_actual_summary or "").strip()

        recent_summaries = _format_rolling_context(new_two_sections_ago, new_prev)

        return {
            # ── Section metadata ──────────────────────────────────────
            "current_section_id":              section["id"],
            "current_section_title":           section["title"],
            "current_section_plan":            section["plan"],
            "current_section_position":        section["position"],
            "current_section_runner":          section["runner"],
            # ── Reset per-section transient vars ──────────────────────
            "current_section_sources":         "",
            "current_section_actual_summary":  "",
            # ── Updated rolling context ───────────────────────────────
            "prev_section_summary":            new_prev,
            "two_sections_ago_summary":        new_two_sections_ago,
            "recent_summaries":                recent_summaries,
            # ── Trimmed queue ─────────────────────────────────────────
            "section_queue":                   json.dumps(queue),
        }

    def __repr__(self) -> str:
        return f"SectionContextTool(name={self.name!r})"
