"""Convert the sources, split them by article, chunk, embed and store — once."""

import json
import os
import re
from pathlib import Path

from yait_aichain.tools import chunk, embed
from yait_aichain.utilities import Vectors

from convert import convert

DATA = Path(__file__).parent / "data"
#: An article heading on a line of its own; the PDF text keeps the double
#: spaces of the printed layout ("Article  83").
ARTICLE = re.compile(r"(?m)^Article\s+(\d+)\s*$")
#: The annexes follow the last article and quote article numbers as headings.
ANNEX = re.compile(r"(?m)^ANNEX\b")
NAMES = {"gdpr": "GDPR", "ai_act": "AI Act", "data_act": "Data Act", "scan": "Privacy Act of 1974"}
SCAN_ARTICLE = "S. 3418, page 7"
#: Anthropic has no embeddings; one word changes the provider — "openai" with
#: text-embedding-3-small, "voyage", "cohere" — and the dimension with it.
EMBEDDER, DIMENSION = ("google", "gemini-embedding-001"), 1536


def articles(source: str, text: str) -> list[dict]:
    """The enacting terms, one part per article; recitals and annexes are left out."""
    if source == "scan":
        return [{"regulation": NAMES[source], "article": SCAN_ARTICLE, "text": " ".join(text.split())}]
    body = text[ARTICLE.search(text).start():]
    if annex := ANNEX.search(body):
        body = body[:annex.start()]
    heads = list(ARTICLE.finditer(body))
    ends = [h.start() for h in heads[1:]] + [len(body)]
    return [{"regulation": NAMES[source], "article": h.group(1), "text": " ".join(body[h.end():end].split())}
            for h, end in zip(heads, ends)]


def converted(sources: dict, receipt) -> list[dict]:
    """Every article of every source, converting each file only the first time."""
    parts, cache = [], DATA / "converted"
    cache.mkdir(parents=True, exist_ok=True)
    for source, path in sources.items():
        saved = cache / f"{source}.json"
        if not saved.exists():
            result = convert.run(path=str(path))
            if "usage" in result:                     # a scanned page, read by a vision model
                receipt.add(result["usage"])
            saved.write_text(json.dumps(result))
        parts += articles(source, json.loads(saved.read_text())["text"])
    return parts


def build(sources: dict, receipt, refresh: bool = False) -> tuple[Vectors, list[dict]]:
    with receipt.stage("convert"):
        parts = converted(sources, receipt)
    records = [{"id": f"{p['regulation']}-{p['article']}-{i}".replace(" ", "_"), "text": piece["text"],
                "metadata": {"regulation": p["regulation"], "article": p["article"]}}
               for p in parts for i, piece in enumerate(chunk(max_chars=1500).run(p["text"]))]
    kb = Vectors("chroma", collection="eu-regulations",
                 description="three EU regulations — GDPR, the AI Act and the Data Act — and a page of "
                             "the US Privacy Act of 1974",
                 embedder=receipt.watch(embed(EMBEDDER[0], model=EMBEDDER[1], dimensions=DIMENSION)),
                 url=os.getenv("CHROMA_URL", "http://localhost:8000"))
    with receipt.stage("index"):
        kb.create(dimension=DIMENSION)
        if refresh or kb.count() != len(records):
            if kb.count():
                kb.delete(filter={"regulation": {"$in": list(NAMES.values())}})
            kb.upsert(records)
    return kb, parts
