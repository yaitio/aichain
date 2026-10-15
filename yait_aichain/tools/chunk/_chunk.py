"""``Chunk`` — a text cut into pieces small enough to embed.

Plain text and Markdown, with headings, code blocks and tables handled on
their own terms; each piece is ``{"text", "chars", "metadata": {"headings",
"is_code"}}``, with ``chars <= max_chars``. See docs/tools/retrieval.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .._base import Tool


# ---------------------------------------------------------------------------
# Internal chunk representation
# ---------------------------------------------------------------------------

@dataclass
class _Chunk:
    text:     str
    headings: list[str] = field(default_factory=list)
    is_code:  bool      = False
    #: "text", "table" or "code". Overlap joins text to text only, so a
    #: table's header row is never glued to a data row, nor a section's first
    #: chunk to the end of the section before it.
    kind:     str       = "text"


# ---------------------------------------------------------------------------
# Markdown detection
# ---------------------------------------------------------------------------

_MD_FENCE_RE  = re.compile(r'```|~~~')
_MD_TABLE_RE  = re.compile(r'\|\s*[-:]+\s*\|')
# Emphasis as a pair around words, not a lone marker: one `_` in a CSV
# column name must not make the whole document Markdown.
_MD_INLINE_RE = re.compile(r'\*\*\S[^\n]*?\*\*|__\S[^\n]*?__|(?<![\w*])\*\S[^*\n]*\*(?![\w*])')

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
    return [(l, t) for l, t in stack if l < level] + [(level, text)]


def _headings_list(stack: list[tuple[int, str]]) -> list[str]:
    return [t for _, t in stack]


# ---------------------------------------------------------------------------
# Text splitting helpers
# ---------------------------------------------------------------------------

def _force_split(text: str, max_chars: int) -> list[str]:
    """Cut by characters: the last resort, for one token longer than the limit."""
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def _split_by_priority(
    text:      str,
    max_chars: int,
    seps:      list[str],
) -> list[str]:
    """Split *text* into pieces of at most *max_chars* on the first of *seps*
    that splits it, recursing with the rest on a piece still too large."""
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

    # No separator produced a split: one token longer than the limit.
    return _force_split(text, max_chars)


# Chinese and Japanese end a sentence without a space after it. A line break
# splits plain text too.
_CJK_SEPS        = ["。", "！", "？"]
_MD_TEXT_SEPS    = ["\n\n", "\n", ". ", *_CJK_SEPS, " "]
_PLAIN_TEXT_SEPS = ["\n\n", "\n", ". ", *_CJK_SEPS, " "]


# ---------------------------------------------------------------------------
# Code blocks and tables: lines packed under a repeated header
# ---------------------------------------------------------------------------

def _pack(prefix: str, lines: list[str], suffix: str, max_chars: int,
          is_code: bool = False) -> list[_Chunk]:
    """Pack *lines* into chunks of at most *max_chars*, each wrapped in *prefix*
    and *suffix* (fences, header rows). A line over the budget is cut by
    characters: the size limit matters more than a well-formed line."""
    budget = max(1, max_chars - len(prefix) - len(suffix))
    chunks: list[_Chunk] = []
    buf: list[str] = []

    def flush(rows: list[str]) -> None:
        if rows:
            chunks.append(_Chunk(text=prefix + "\n".join(rows) + suffix, is_code=is_code,
                                 kind="code" if is_code else "table"))

    for line in lines:
        if len(line) > budget:
            flush(buf)
            buf = []
            for piece in _force_split(line, budget):
                flush([piece])
        elif len(prefix + "\n".join(buf + [line]) + suffix) <= max_chars:
            buf.append(line)
        else:
            flush(buf)
            buf = [line]
    flush(buf)
    return chunks


def _chunk_code(header: str, body: str, fence: str, max_chars: int) -> list[_Chunk]:
    """A fenced code block, split by lines with the opening line and the
    fence repeated in every piece."""
    full = f"{header}\n{body}\n{fence}"
    if len(full) <= max_chars:
        return [_Chunk(text=full, is_code=True, kind="code")]
    return _pack(f"{header}\n", body.split("\n"), f"\n{fence}", max_chars, is_code=True)


_SEP_ROW_RE = re.compile(r'^\s*\|[\s\-:|]+\|')


def _chunk_table(table: str, max_chars: int) -> list[_Chunk]:
    """A Markdown table, split by rows with the header rows and the separator
    row repeated in every piece, so each is a valid table."""
    if len(table) <= max_chars:
        return [_Chunk(text=table, kind="table")]
    rows = table.split("\n")
    sep_idx = next((i for i, r in enumerate(rows) if _SEP_ROW_RE.match(r)), None)
    header = "\n".join(rows[:sep_idx + 1]) if sep_idx is not None else ""
    # Not a table, or a header that leaves no room for a row: split as text.
    if sep_idx is None or len(header) + 2 >= max_chars:
        return [_Chunk(text=p.rstrip(), kind="table")
                for p in _split_by_priority(table, max_chars, _MD_TEXT_SEPS)]
    return _pack(header + "\n", [r for r in rows[sep_idx + 1:] if r.strip()], "", max_chars)


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


_FENCE_RE   = re.compile(r'^(`{3,}|~{3,})')
_HEADING_RE = re.compile(r'^(#{1,6})\s+(.*)')
_ROW_RE     = re.compile(r'^\s*\|')


def _starts_block(line: str) -> bool:
    """True when *line* opens a code block, a heading or a table. One definition
    for the parser and the text loop: if they disagreed, the outer loop would spin."""
    return bool(_FENCE_RE.match(line) or _HEADING_RE.match(line) or _ROW_RE.match(line))


def _parse_markdown(text: str) -> list[_Segment]:
    """*text* as a flat list of typed segments, in one pass; an unclosed fence is text."""
    lines:    list[str]      = text.split("\n")
    segments: list[_Segment] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # ── Fenced code block ─────────────────────────────────────────
        fence_m = _FENCE_RE.match(line)
        if fence_m:
            fence_chars = fence_m.group(1)
            header      = line
            body_lines: list[str] = []
            i += 1
            closed = False
            # A closing fence is the fence character alone, at least as many
            # as opened it (CommonMark): a "```python" line inside a block
            # does not close it.
            closing = re.compile(rf"^ {{0,3}}{re.escape(fence_chars[0])}{{{len(fence_chars)},}}\s*$")
            while i < n:
                if closing.match(lines[i]):
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
        head_m = _HEADING_RE.match(line)
        if head_m:
            segments.append(_Segment(
                kind    = "heading",
                content = head_m.group(2).strip(),
                level   = len(head_m.group(1)),
            ))
            i += 1
            continue

        # ── Table ─────────────────────────────────────────────────────
        if _ROW_RE.match(line):
            tbl_lines: list[str] = []
            while i < n and _ROW_RE.match(lines[i]):
                tbl_lines.append(lines[i])
                i += 1
            if tbl_lines:
                segments.append(_Segment(kind="table", content="\n".join(tbl_lines)))
            continue

        # ── Plain text ────────────────────────────────────────────────
        txt_lines: list[str] = []
        while i < n and not _starts_block(lines[i]):
            txt_lines.append(lines[i])
            i += 1
        content = "\n".join(txt_lines)
        if content.strip():
            segments.append(_Segment(kind="text", content=content))

    return segments


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------

class Chunk(Tool):
    """
    Split text into overlapping chunks for embedding and vector-store ingestion.

    Handles plain text and Markdown with dedicated logic for headings, code
    blocks, and tables.  Output is a list of dicts ready for
    ``Vectors.upsert`` once each has an ``"id"``.

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
        When ``overlap_chars >= max_chars`` at construction, or when a call
        passes ``max_chars``, ``overlap_chars`` or ``merge_peers`` — they are
        settings of the tool.

    Examples
    --------
    ::

        from yait_aichain.tools import chunk

        chunker = chunk(max_chars=800, overlap_chars=80)
        chunks  = chunker.run(markdown_text)
        # chunks[0] == {"text": "...", "chars": 247, "metadata": {"headings": ["Intro"], "is_code": False}}
    """

    risk = "read"                        # it splits text; it reaches nothing
    _safe_for_untrusted_load = True

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
        },
        "required": ["input"],
    }

    def __init__(
        self,
        max_chars:     int  = 1500,
        overlap_chars: int | None = None,
        merge_peers:   bool = True,
    ) -> None:
        # A tenth of a piece unless given: 150 at the default 1 500. A fixed
        # 150 cut 780 characters into 30 near-identical windows at
        # max_chars=200 — every piece three-quarters its neighbour.
        if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
            raise ValueError(f"max_chars is a whole number of at least 1, not {max_chars!r}")
        if overlap_chars is None:
            overlap_chars = max_chars // 10
        if isinstance(overlap_chars, bool) or not isinstance(overlap_chars, int) or overlap_chars < 0:
            # A negative overlap broke the size contract: pieces came out
            # longer than max_chars.
            raise ValueError(f"overlap_chars is a whole number of 0 or more, not {overlap_chars!r}")
        if overlap_chars >= max_chars:
            raise ValueError(
                f"overlap_chars ({overlap_chars}) must be less than "
                f"max_chars ({max_chars})."
            )
        self._max_chars      = max_chars
        self._overlap_chars  = overlap_chars
        self._merge_enabled  = merge_peers

    def _serialise_init_args(self) -> dict:
        """What `Chain.save` writes: the sizes, which decide how a store was filled."""
        return {"max_chars": self._max_chars, "overlap_chars": self._overlap_chars,
                "merge_peers": self._merge_enabled}

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
            Takes no keys. ``max_chars``, ``overlap_chars`` and
            ``merge_peers`` are settings of the tool and raise here.

        Returns
        -------
        list[dict]
            Each item: ``{"text": str, "chars": int, "metadata": {"headings": list[str], "is_code": bool}}``
        """
        # The sizes are the tool's settings: a store is filled with pieces cut
        # one way. Any other key is a typo nothing reads.
        if options and (given := [k for k in ("max_chars", "overlap_chars", "merge_peers") if k in options]):
            raise ValueError(f"{self.name}: {', '.join(given)} is a setting of the chunker, not of a call "
                             f"— chunk({given[0]}=…)")
        if options:
            raise ValueError(f"{self.name} takes no options; got {sorted(options)}")
        if not input or not input.strip():
            return []
        # Windows line ends: "\r\n\r\n" is a paragraph break too.
        input = input.replace("\r\n", "\n").replace("\r", "\n")
        max_chars     = self._max_chars
        overlap_chars = self._overlap_chars
        merge_peers   = self._merge_enabled

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
                    kind     = "text" if prev.kind == chunk.kind == "text" else "table",
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
        """Prepend the tail of each text chunk to the next one under the same
        headings, capped at *max_chars* and starting at a word boundary."""
        if not chunks or overlap_chars == 0:
            return chunks

        result: list[_Chunk] = [chunks[0]]

        for prev, chunk in zip(chunks, chunks[1:]):
            # Text to text in one section only, and the tail is the neighbour's
            # own text — not what overlap already added to it.
            if prev.kind != "text" or chunk.kind != "text" or prev.headings != chunk.headings:
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
                kind     = chunk.kind,
            ))

        return result

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Chunk(max_chars={self._max_chars}, "
            f"overlap_chars={self._overlap_chars}, "
            f"merge_peers={self._merge_enabled})"
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

