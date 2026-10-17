"""The two 10-K filings, split by Item, chunked, embedded and stored — once.
A copy of cite's index.py, split by Item instead of Article."""

import json
import os
import re
from pathlib import Path

from yait_aichain.tools import chunk, embed
from yait_aichain.utilities import Vectors

from convert import convert

DATA = Path(__file__).parent / "data"
#: A 10-K Item heading on a line of its own: "Item 7.    Management’s Discussion…".
ITEM = re.compile(r"(?m)^Item\s+(\d+[A-C]?)\.\s+(.+)$")
FILINGS = {"10k_fy2025": "10-K FY2025", "10k_fy2024": "10-K FY2024"}
EMBEDDER, DIMENSION = ("google", "gemini-embedding-001"), 1536


def items(filing: str, text: str) -> list[dict]:
    """One part per Item, from the first Item heading on."""
    heads = list(ITEM.finditer(text))
    ends = [h.start() for h in heads[1:]] + [len(text)]
    return [{"filing": FILINGS[filing], "item": h.group(1), "title": " ".join(h.group(2).split()),
             "text": " ".join(text[h.end():end].split())} for h, end in zip(heads, ends)]


def converted(sources: dict) -> list[dict]:
    parts, cache = [], DATA / "converted"
    cache.mkdir(parents=True, exist_ok=True)
    for source in FILINGS:
        saved = cache / f"{source}.json"
        if not saved.exists():
            saved.write_text(json.dumps(convert.run(path=str(sources[source]))))
        parts += items(source, json.loads(saved.read_text())["text"])
    return parts


def build(sources: dict, receipt, refresh: bool = False) -> tuple[Vectors, list[dict]]:
    with receipt.stage("convert"):
        parts = converted(sources)
    records = [{"id": f"{p['filing']}-{p['item']}-{i}".replace(" ", "_"), "text": piece["text"],
                "metadata": {"filing": p["filing"], "item": p["item"], "title": p["title"]}}
               for p in parts for i, piece in enumerate(chunk(max_chars=1500).run(p["text"]))]
    kb = Vectors("chroma", collection="apple-10k",
                 description="Apple's annual reports on Form 10-K for fiscal 2025 and fiscal 2024",
                 embedder=receipt.watch(embed(EMBEDDER[0], model=EMBEDDER[1], dimensions=DIMENSION)),
                 url=os.getenv("CHROMA_URL", "http://localhost:8000"))
    with receipt.stage("index"):
        kb.create(dimension=DIMENSION)
        if refresh or kb.count() != len(records):
            if kb.count():
                kb.delete(filter={"filing": {"$in": list(FILINGS.values())}})
            kb.upsert(records)
    return kb, parts
