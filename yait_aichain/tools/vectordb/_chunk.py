"""
tools.vectordb._chunk
======================

VectorChunkTool — intelligent text chunker for RAG pipelines.

Splits long text into overlapping chunks suitable for embedding and
vector-store ingestion.  Supports plain text and Markdown with dedicated
handling for headings, code blocks, and tables.

Output schema (per chunk)
--------------------------
::

    {
        "text":  str,     # chunk content (may include overlap prefix)
        "chars": int,     # len(text)  — always <= max_chars
        "metadata": {
            "headings": list[str],   # heading path at this chunk's position
            "is_code":  bool,        # True for code-block chunks
        }
    }

Format detection
----------------
Text is treated as Markdown when it contains at least one of:
  • A line starting with ``#``
  • A fenced code block marker (`` ``` `` or ``~~~``)
  • A table separator row  (``| --- |``)
  • Inline emphasis markers (``**``, ``__``, ``*``, ``_``)

Markdown chunking
-----------------
Segments are parsed in a single pass:

  Headings   — update the heading stack; never emitted as chunks.
               When heading level N appears, all stack entries with
               level ≥ N are replaced by the new heading.

  Code blocks — kept whole if ≤ max_chars; otherwise split line-by-line
                with the opening fence repeated in every sub-chunk.

  Tables      — kept whole if ≤ max_chars; otherwise split row-by-row
                with the header + separator rows repeated.

  Text        — split by priority: \\n\\n → \\n → ". " → space.

Plain text
----------
Split priority: \\n\\n → ". " → space.
``metadata.headings`` is always ``[]``.

Overlap
-------
The last ``overlap_chars`` characters of chunk N are prepended to chunk N+1.
The left boundary of the overlap window is shifted right to the nearest space
so that no word is split across the boundary.  The final prepend is trimmed to
ensure ``chunk["chars"] <= max_chars``.  Overlap is never applied to code chunks.

Merge peers
-----------
When ``merge_peers=True`` (default) adjacent non-code chunks that share the
same heading path are merged if their combined length fits in ``max_chars``.

Edge cases
----------
  empty string          → ``[]``
  headings only         → ``[]``
  unclosed code block   → treated as plain text to end of input
  overlap_chars ≥ max_chars → ``ValueError`` on construction and on run()

Chain usage
-----------
::

    from tools.vectordb import VectorDB, vectorChunk, vectorUpsert
    from tools.embedding import Embedding

    chunker = vectorChunk(max_chars=800, overlap_chars=80)
    store   = VectorDB("chroma", "docs",
                        embedder=Embedding("openai/text-embedding-3-small"))
    upserter = vectorUpsert(store)

    chunks  = chunker.run(long_markdown_text)
    records = [{"id": f"doc_{i}", **c} for i, c in enumerate(chunks)]
    upserter.run(records)

    # Or inside a Chain — chunker output feeds directly into vectorUpsert
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .._base import Tool


# ---------------------------------------------------------------------------
# Internal chunk representation
# ---------------------------------------------------------------------------

@dataclass
class _Chunk:
    text:     str
    headings: list[str] = field(default_factory=list)
    is_code:  bool      = False


# ---------------------------------------------------------------------------
# Markdown detection
# ---------------------------------------------------------------------------

_MD_FENCE_RE  = re.compile(r'```|~~~')
_MD_TABLE_RE  = re.compile(r'\|\s*[-:]+\s*\|')
_MD_INLINE_RE = re.compile(r'\*\*|__|\*|(?<!_)_(?!_)')

def _is_markdown(text: str) -> bool:
    for line in text.splitlines():
        if line.startswith('#'):
            return True
    if _MD_FENCE_RE.search(text):
        return True
    if _MD_TABLE_RE.search(text):
        return True
    if _MD_INLINE_RE.search(text):
        return True
    return False


# ---------------------------------------------------------------------------
# Heading stack helpers
# ---------------------------------------------------------------------------

def _push_heading(
    stack: list[tuple[int, str]],
    level: int,
    text:  str,
) -> list[tuple[int, str]]:
    """
    Update heading stack for a new heading of *level*.

    All entries with level >= *level* are dropped (they are children of the
    incoming heading's parent), then the new heading is appended.
    """
    return [(l, t) for l, t in stack if l < level] + [(level, text)]


def _headings_list(stack: list[tuple[int, str]]) -> list[str]:
    return [t for _, t in stack]


# ---------------------------------------------------------------------------
# Text splitting helpers
# ---------------------------------------------------------------------------

def _force_split(text: str, max_chars: int) -> list[str]:
    """
    Split *text* at word boundaries into pieces of at most *max_chars*.

    A single token longer than *max_chars* is hard-split at the character
    level as a last resort.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    buf = ""

    for word in text.split(" "):
        candidate = (buf + " " + word) if buf else word
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            if len(word) > max_chars:
                # Hard-split oversized token
                for i in range(0, len(word), max_chars):
                    chunks.append(word[i : i + max_chars])
                buf = ""
            else:
                buf = word

    if buf:
        chunks.append(buf)
    return chunks


def _split_by_priority(
    text:      str,
    max_chars: int,
    seps:      list[str],
) -> list[str]:
    """
    Recursively split *text* into pieces ≤ *max_chars* using *seps* in order.

    Each separator is tried in turn; the first one that produces multiple
    parts drives the split.  When no separator applies, falls back to
    word-boundary splitting via :func:`_force_split`.
    """
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    for idx, sep in enumerate(seps):
        raw_parts = text.split(sep)
        if len(raw_parts) == 1:
            continue  # separator absent — try next

        # Keep each separator attached to the part that precedes it, so no
        # text is lost at chunk boundaries (e.g. the trailing "." of a
        # sentence when splitting on ". ").  Concatenating the parts back
        # reproduces the input exactly.
        parts = [p + sep for p in raw_parts[:-1]] + [raw_parts[-1]]

        chunks: list[str] = []
        buf = ""

        for part in parts:
            candidate = buf + part
            if len(candidate) <= max_chars:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                if len(part) <= max_chars:
                    buf = part
                else:
                    # Part still too large — recurse with remaining separators
                    sub = _split_by_priority(part, max_chars, seps[idx + 1 :])
                    if sub:
                        chunks.extend(sub[:-1])
                        buf = sub[-1]
                    else:
                        buf = ""

        if buf:
            chunks.append(buf)

        return [c for c in chunks if c.strip()]

    # No separator produced a split — fall back to word boundaries
    return _force_split(text, max_chars)


_MD_TEXT_SEPS    = ["\n\n", "\n", ". ", " "]
_PLAIN_TEXT_SEPS = ["\n\n", ". ", " "]


# ---------------------------------------------------------------------------
# Code block chunking
# ---------------------------------------------------------------------------

def _chunk_code(
    header:    str,
    body:      str,
    fence:     str,
    max_chars: int,
) -> list[_Chunk]:
    """
    Chunk a fenced code block.

    Parameters
    ----------
    header : str   — opening fence line, e.g. ````` ```python `````
    body   : str   — code content between fences
    fence  : str   — fence character sequence (````` ``` ````` or ``~~~``)
    """
    full = f"{header}\n{body}\n{fence}"
    if len(full) <= max_chars:
        return [_Chunk(text=full, is_code=True)]

    # Split line by line; repeat header + fence in every sub-chunk
    lines:     list[str]   = body.split("\n")
    chunks:    list[_Chunk] = []
    buf_lines: list[str]   = []

    # Budget for code content inside one chunk (header + body + fence and
    # the two joining newlines must all fit into max_chars).
    line_budget = max(1, max_chars - len(header) - len(fence) - 2)

    def _flush(lines_: list[str]) -> None:
        if lines_:
            chunks.append(_Chunk(
                text    = f"{header}\n" + "\n".join(lines_) + f"\n{fence}",
                is_code = True,
            ))

    for line in lines:
        if len(line) > line_budget:
            # A single line longer than the whole budget: hard-wrap it,
            # otherwise the chunk would break the `chars <= max_chars`
            # contract (and downstream metadata/token limits with it).
            _flush(buf_lines)
            buf_lines = []
            for i in range(0, len(line), line_budget):
                _flush([line[i : i + line_budget]])
            continue
        candidate = f"{header}\n" + "\n".join(buf_lines + [line]) + f"\n{fence}"
        if len(candidate) <= max_chars:
            buf_lines.append(line)
        else:
            _flush(buf_lines)
            buf_lines = [line]

    _flush(buf_lines)
    return chunks


# ---------------------------------------------------------------------------
# Table chunking
# ---------------------------------------------------------------------------

_SEP_ROW_RE = re.compile(r'^\s*\|[\s\-:|]+\|')


def _chunk_table(table: str, max_chars: int) -> list[_Chunk]:
    """
    Chunk a Markdown table.

    Keeps header rows + separator row in every sub-chunk so each piece is a
    valid, self-contained table.
    """
    if len(table) <= max_chars:
        return [_Chunk(text=table)]

    rows    = table.split("\n")
    sep_idx = next((i for i, r in enumerate(rows) if _SEP_ROW_RE.match(r)), None)

    if sep_idx is None:
        # Not a proper table — fall back to text splitting
        pieces = _split_by_priority(table, max_chars, _MD_TEXT_SEPS)
        return [_Chunk(text=p.rstrip()) for p in pieces]

    header_rows  = rows[:sep_idx]
    sep_row      = rows[sep_idx]
    data_rows    = rows[sep_idx + 1 :]
    header_block = "\n".join(header_rows) + "\n" + sep_row

    # If the header block alone leaves no room for even one data row, repeating
    # it in every sub-chunk would break the `chars <= max_chars` contract.
    # Fall back to plain size-based splitting of the whole table instead.
    if len(header_block) + 2 >= max_chars:
        pieces = _split_by_priority(table, max_chars, _MD_TEXT_SEPS)
        return [_Chunk(text=p.rstrip()) for p in pieces]

    chunks:   list[_Chunk] = []
    buf_rows: list[str]    = []

    def _flush_table(rows_: list[str]) -> None:
        if rows_:
            chunks.append(_Chunk(text=header_block + "\n" + "\n".join(rows_)))

    # Budget for data rows inside one chunk (header block + newline + rows).
    row_budget = max(1, max_chars - len(header_block) - 1)

    for row in data_rows:
        if not row.strip():
            continue
        if len(row) > row_budget:
            # A single row longer than the whole budget: hard-wrap it so the
            # `chars <= max_chars` contract holds.  The wrapped pieces are
            # no longer well-formed table rows, but staying within the size
            # contract matters more (metadata/token limits downstream).
            _flush_table(buf_rows)
            buf_rows = []
            for i in range(0, len(row), row_budget):
                _flush_table([row[i : i + row_budget]])
            continue
        candidate = header_block + "\n" + "\n".join(buf_rows + [row])
        if len(candidate) <= max_chars:
            buf_rows.append(row)
        else:
            _flush_table(buf_rows)
            buf_rows = [row]

    _flush_table(buf_rows)
    return chunks


# ---------------------------------------------------------------------------
# Markdown segment parser
# ---------------------------------------------------------------------------

@dataclass
class _Segment:
    kind:    str        # "heading" | "code" | "table" | "text"
    content: str
    level:   int = 0   # headings only
    header:  str = ""  # code: opening fence line
    fence:   str = ""  # code: fence characters (``` or ~~~)


def _parse_markdown(text: str) -> list[_Segment]:
    """
    Parse *text* into a flat list of typed segments in a single pass.

    Unclosed code fences are emitted as plain-text segments.
    """
    lines:    list[str]      = text.split("\n")
    segments: list[_Segment] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # ── Fenced code block ─────────────────────────────────────────
        fence_m = re.match(r'^(`{3,}|~{3,})', line)
        if fence_m:
            fence_chars = fence_m.group(1)
            header      = line
            body_lines: list[str] = []
            i += 1
            closed = False
            while i < n:
                if lines[i].startswith(fence_chars):
                    closed = True
                    i += 1
                    break
                body_lines.append(lines[i])
                i += 1
            if closed:
                segments.append(_Segment(
                    kind    = "code",
                    content = "\n".join(body_lines),
                    header  = header,
                    fence   = fence_chars,
                ))
            else:
                # Unclosed — treat as plain text
                segments.append(_Segment(
                    kind    = "text",
                    content = header + "\n" + "\n".join(body_lines),
                ))
            continue

        # ── ATX heading ───────────────────────────────────────────────
        head_m = re.match(r'^(#{1,6})\s+(.*)', line)
        if head_m:
            segments.append(_Segment(
                kind    = "heading",
                content = head_m.group(2).strip(),
                level   = len(head_m.group(1)),
            ))
            i += 1
            continue

        # ── Table ─────────────────────────────────────────────────────
        if re.match(r'^\s*\|', line):
            tbl_lines: list[str] = []
            while i < n and re.match(r'^\s*\|', lines[i]):
                tbl_lines.append(lines[i])
                i += 1
            if tbl_lines:
                segments.append(_Segment(kind="table", content="\n".join(tbl_lines)))
            continue

        # ── Plain text ────────────────────────────────────────────────
        txt_lines: list[str] = []
        while i < n:
            l = lines[i]
            if (re.match(r'^(`{3,}|~{3,})', l)
                    or re.match(r'^#{1,6}\s', l)
                    or re.match(r'^\s*\|', l)):
                break
            txt_lines.append(l)
            i += 1
        content = "\n".join(txt_lines)
        if content.strip():
            segments.append(_Segment(kind="text", content=content))

    return segments


# ---------------------------------------------------------------------------
# VectorChunkTool
# ---------------------------------------------------------------------------

class VectorChunkTool(Tool):
    """
    Split text into overlapping chunks for embedding and vector-store ingestion.

    Handles plain text and Markdown with dedicated logic for headings, code
    blocks, and tables.  Output is a list of dicts ready to pass directly to
    :class:`VectorUpsertTool` (after adding an ``"id"`` field).

    Parameters
    ----------
    max_chars : int
        Maximum characters per chunk (default 1500).
    overlap_chars : int
        Overlap in characters between consecutive non-code chunks (default 150).
    merge_peers : bool
        Merge adjacent non-code chunks with identical heading paths if they
        fit within ``max_chars`` together (default True).

    Raises
    ------
    ValueError
        When ``overlap_chars >= max_chars`` at construction or at ``run()``
        time (when overridden via options).

    Examples
    --------
    ::

        from tools.vectordb import vectorChunk

        chunker = vectorChunk(max_chars=800, overlap_chars=80)
        chunks  = chunker.run(markdown_text)
        # chunks[0] == {"text": "...", "chars": 247, "metadata": {"headings": ["Intro"], "is_code": False}}
    """

    name        = "vector_chunk"
    description = (
        "Split text into overlapping chunks for embedding and vector-store "
        "ingestion.  Handles plain text and Markdown, preserving headings, "
        "code blocks, and tables intact where possible."
    )
    parameters  = {
        "type": "object",
        "properties": {
            "input": {
                "type":        "string",
                "description": "The text to chunk.",
            },
            "options": {
                "type":        "object",
                "description": "Chunking configuration.",
                "properties": {
                    "max_chars": {
                        "type":        "integer",
                        "description": "Maximum characters per chunk (default 1500).",
                    },
                    "overlap_chars": {
                        "type":        "integer",
                        "description": "Overlap in characters between consecutive chunks (default 150).",
                    },
                    "merge_peers": {
                        "type":        "boolean",
                        "description": (
                            "Merge adjacent small chunks with identical headings "
                            "(default true)."
                        ),
                    },
                },
            },
        },
        "required": ["input"],
    }

    def __init__(
        self,
        max_chars:     int  = 1500,
        overlap_chars: int  = 150,
        merge_peers:   bool = True,
    ) -> None:
        if overlap_chars >= max_chars:
            raise ValueError(
                f"overlap_chars ({overlap_chars}) must be less than "
                f"max_chars ({max_chars})."
            )
        self._max_chars      = max_chars
        self._overlap_chars  = overlap_chars
        self._merge_enabled  = merge_peers

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        input:   str,
        options: dict | None = None,
    ) -> list[dict]:
        """
        Chunk *input* text and return a list of chunk dicts.

        Parameters
        ----------
        input : str
            Text to chunk.  Returns ``[]`` for empty or whitespace-only input.
        options : dict | None
            Per-call overrides for ``max_chars``, ``overlap_chars``,
            ``merge_peers``.

        Returns
        -------
        list[dict]
            Each item: ``{"text": str, "chars": int, "metadata": {"headings": list[str], "is_code": bool}}``
        """
        if not input or not input.strip():
            return []

        opts          = options or {}
        max_chars     = int(opts.get("max_chars",     self._max_chars))
        overlap_chars = int(opts.get("overlap_chars", self._overlap_chars))
        merge_peers   = bool(opts.get("merge_peers",  self._merge_enabled))

        if overlap_chars >= max_chars:
            raise ValueError(
                f"overlap_chars ({overlap_chars}) must be less than "
                f"max_chars ({max_chars})."
            )

        # ── Chunk ────────────────────────────────────────────────────────
        # Reserve room for the overlap tail up front.  Splitting to the full
        # max_chars would leave no headroom, and _apply_overlap would then
        # silently skip every chunk — overlap must shrink the split budget,
        # not compete with it.
        effective_max = max_chars - overlap_chars
        chunks = (
            self._chunk_markdown(input, effective_max)
            if _is_markdown(input)
            else self._chunk_plain(input, effective_max)
        )

        if not chunks:
            return []

        # ── Merge peers ──────────────────────────────────────────────────
        if merge_peers:
            chunks = self._merge_peers(chunks, effective_max)

        # ── Apply overlap ────────────────────────────────────────────────
        chunks = self._apply_overlap(chunks, overlap_chars, max_chars)

        # ── Serialise ────────────────────────────────────────────────────
        return [
            {
                "text":  c.text,
                "chars": len(c.text),
                "metadata": {
                    "headings": list(c.headings),
                    "is_code":  c.is_code,
                },
            }
            for c in chunks
        ]

    # ── Markdown chunking ─────────────────────────────────────────────────────

    def _chunk_markdown(self, text: str, max_chars: int) -> list[_Chunk]:
        segments:      list[_Segment]       = _parse_markdown(text)
        heading_stack: list[tuple[int, str]] = []
        chunks:        list[_Chunk]          = []

        for seg in segments:
            if seg.kind == "heading":
                heading_stack = _push_heading(heading_stack, seg.level, seg.content)
                continue

            headings = _headings_list(heading_stack)

            if seg.kind == "code":
                for c in _chunk_code(seg.header, seg.content, seg.fence, max_chars):
                    c.headings = list(headings)
                    chunks.append(c)

            elif seg.kind == "table":
                for c in _chunk_table(seg.content, max_chars):
                    c.headings = list(headings)
                    chunks.append(c)

            else:  # text
                for p in _split_by_priority(seg.content, max_chars, _MD_TEXT_SEPS):
                    # Separators stay attached during splitting (lossless);
                    # trailing whitespace is cosmetic at this point.
                    chunks.append(_Chunk(text=p.rstrip(), headings=list(headings)))

        return chunks

    # ── Plain text chunking ───────────────────────────────────────────────────

    def _chunk_plain(self, text: str, max_chars: int) -> list[_Chunk]:
        return [
            _Chunk(text=p.rstrip(), headings=[])
            for p in _split_by_priority(text, max_chars, _PLAIN_TEXT_SEPS)
        ]

    # ── Merge peers ───────────────────────────────────────────────────────────

    def _merge_peers(self, chunks: list[_Chunk], max_chars: int) -> list[_Chunk]:
        if not chunks:
            return chunks
        merged: list[_Chunk] = [chunks[0]]
        for chunk in chunks[1:]:
            prev = merged[-1]
            if (
                prev.headings == chunk.headings
                and not prev.is_code
                and not chunk.is_code
                and len(prev.text) + 1 + len(chunk.text) <= max_chars
            ):
                merged[-1] = _Chunk(
                    text     = prev.text + "\n" + chunk.text,
                    headings = list(prev.headings),
                    is_code  = False,
                )
            else:
                merged.append(chunk)
        return merged

    # ── Apply overlap ─────────────────────────────────────────────────────────

    def _apply_overlap(
        self,
        chunks:        list[_Chunk],
        overlap_chars: int,
        max_chars:     int,
    ) -> list[_Chunk]:
        """
        Prepend the tail of each chunk to the next.

        Rules
        -----
        • Skip overlap when either chunk is a code block.
        • The tail length is capped so ``len(new_text) <= max_chars``.
        • The left edge of the tail is shifted right to the nearest space
          so no word is split at the overlap boundary.
        """
        if not chunks or overlap_chars == 0:
            return chunks

        result: list[_Chunk] = [chunks[0]]

        for chunk in chunks[1:]:
            prev = result[-1]

            if prev.is_code or chunk.is_code:
                result.append(chunk)
                continue

            # How much space is available before we'd exceed max_chars?
            available = max_chars - len(chunk.text)
            if available <= 0:
                result.append(chunk)
                continue

            window = min(overlap_chars, available)
            tail   = prev.text[-window:] if len(prev.text) > window else prev.text

            # Shift the left boundary of the tail right to the first space
            # so we don't start the overlap mid-word.
            space_idx = tail.find(" ")
            if space_idx > 0:
                tail = tail[space_idx + 1:]

            if tail:
                new_text = tail + " " + chunk.text
                # Final safety trim (guards against edge cases)
                if len(new_text) > max_chars:
                    new_text = chunk.text
            else:
                new_text = chunk.text

            result.append(_Chunk(
                text     = new_text,
                headings = list(chunk.headings),
                is_code  = chunk.is_code,
            ))

        return result

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"VectorChunkTool(max_chars={self._max_chars}, "
            f"overlap_chars={self._overlap_chars}, "
            f"merge_peers={self._merge_enabled})"
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def vectorChunk(
    max_chars:     int  = 1500,
    overlap_chars: int  = 150,
    merge_peers:   bool = True,
) -> VectorChunkTool:
    """
    Return a configured :class:`VectorChunkTool`.

    Parameters
    ----------
    max_chars     : int  — maximum characters per chunk (default 1500)
    overlap_chars : int  — overlap in characters (default 150)
    merge_peers   : bool — merge adjacent small chunks with identical headings

    Examples
    --------
    ::

        from tools.vectordb import vectorChunk

        chunker = vectorChunk(max_chars=800, overlap_chars=80)
        chunks  = chunker.run(my_text)
    """
    return VectorChunkTool(
        max_chars     = max_chars,
        overlap_chars = overlap_chars,
        merge_peers   = merge_peers,
    )
