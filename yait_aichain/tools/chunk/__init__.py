"""
tools.chunk
===========

A long text cut into pieces small enough to embed, along its own structure —
headings, paragraphs, code and tables kept whole where they fit. ``chunk(…)``
makes the tool; until 3.5.0 it lived in ``tools.vectordb`` as
``vectorChunk``, which still works.
"""

from ._chunk import VectorChunkTool, vectorChunk


def chunk(*, max_chars: int = 1500, overlap_chars: int = 150, merge_peers: bool = True,
          name: "str | None" = None, description: "str | None" = None) -> VectorChunkTool:
    """
    A chunking tool. The sizes are settings: a store is filled with pieces cut
    one way, so they are fixed for the tool, not chosen per call.

    ::

        pieces = chunk(max_chars=800).run(long_text)    # [{"text", "headings", …}, …]
    """
    from .._callable import named
    return named(VectorChunkTool(max_chars=max_chars, overlap_chars=overlap_chars,
                                 merge_peers=merge_peers), name, description)


__all__ = ["chunk", "VectorChunkTool", "vectorChunk"]

from .._callable import callable_package as _callable_package
_callable_package(__name__, chunk)
