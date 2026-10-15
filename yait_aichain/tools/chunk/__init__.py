"""A long text cut into pieces small enough to embed, along its own structure.
``chunk(…)`` makes the tool, a :class:`Chunk`. See docs/tools/retrieval.md.
"""

from ._chunk import Chunk


def chunk(*, max_chars: int = 1500, overlap_chars: "int | None" = None, merge_peers: bool = True,
          name: "str | None" = None, description: "str | None" = None) -> Chunk:
    """
    A chunking tool. The sizes are settings: a store is filled with pieces cut
    one way, so they are fixed for the tool, not chosen per call.
    ``overlap_chars`` defaults to a tenth of ``max_chars`` — 150 at the
    default size.

    ::

        pieces = chunk(max_chars=800).run(long_text)    # [{"text", "chars", "metadata"}, …]
    """
    from .._callable import named
    return named(Chunk(max_chars=max_chars, overlap_chars=overlap_chars,
                                 merge_peers=merge_peers), name, description)


__all__ = ["chunk", "Chunk"]

from .._callable import callable_package as _callable_package
_callable_package(__name__, chunk)
