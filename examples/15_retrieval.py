"""
15_retrieval.py — Answer from your own texts: chunk, embed, find, rerank, answer with the source.

Keys: OPENAI_API_KEY (embeddings and the answer), DASHSCOPE_API_KEY (rerank). Three texts need no
store; a real corpus goes into Vectors("qdrant" | "chroma" | "pinecone", ...) with the same embedder.
"""

import math

from yait_aichain import Model, Skill
from yait_aichain.tools import chunk, embed, rerank

TEXTS = {
    "returns": "Items can be returned within 30 days of delivery. Refunds reach the card in 5 days.",
    "shipping": "Orders ship from Rotterdam within two working days. Express costs 9 EUR.",
    "warranty": "Electronics carry a two-year warranty. A repair takes about ten days.",
}
pieces = [{"id": f"{doc}-{i}", "text": p["text"]} for doc, text in TEXTS.items()
          for i, p in enumerate(chunk(max_chars=200).run(text))]

embedder = embed("openai", model="text-embedding-3-small")
vectors = embedder.run([p["text"] for p in pieces]).embeddings
question = "How long until my refund arrives?"
q = embedder.run([question]).embeddings[0]


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.hypot(*a) * math.hypot(*b))


near = sorted(pieces, key=lambda p: -cosine(q, vectors[pieces.index(p)]))[:3]
best = rerank("qwen", model="qwen3-rerank").run([p["text"] for p in near],
                                                  {"query": question, "top_n": 1})[0]
source = near[best["index"]]

# Key: OPENAI_API_KEY from the environment; or Model("gpt-5.4-mini", api_key="...")
answer = Skill(Model("gpt-5.4-mini"), prompt="Answer from this text only.\n\nText: {text}\n\n"
                                             "Question: {question}").run(
    variables={"text": source["text"], "question": question})
print(answer)
print(f"source: {source['id']}")
