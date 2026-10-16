"""Retrieve, rerank and answer with the article that says so; check shape and quotes."""

import os
import re

from yait_aichain import Model, Skill, check_structure

ANSWER_MODEL = os.getenv("ANSWER_MODEL", "claude-sonnet-5-5")
SCHEMA = {"type": "object", "required": ["refused", "answer", "citations"],
          "properties": {
              "refused":   {"type": "boolean"},
              "answer":    {"type": ["string", "null"]},
              "citations": {"type": "array", "items": {
                  "type": "object", "required": ["regulation", "article", "quote"],
                  "properties": {"regulation": {"type": "string"},
                                 "article":    {"type": "string"},
                                 "quote":      {"type": "string"}}}}}}
PROMPT = """Answer the question only from the passages below.
Quote every passage you rely on verbatim, word for word, and give its regulation and
article exactly as the passage is labelled. If the passages do not answer the question,
set refused to true, answer to null, and give no citations.

Question: {question}

Passages:
{passages}"""


def retrieve(kb, question: str, reranker, receipt) -> list[dict]:
    hits = kb.query.run(question, {"n": 20})
    if reranker is None:
        return hits[:6]
    ranked = reranker.run([h["text"] for h in hits], {"query": question, "top_n": 6})
    receipt.add({"cost": None})               # a rerank reports no price
    return [hits[r["index"]] for r in ranked]


def answer(kb, question: str, receipt, model: str = ANSWER_MODEL, reranker=None) -> dict:
    passages = retrieve(kb, question, reranker, receipt)
    labelled = "\n\n".join(f"[{p['metadata']['regulation']} | article {p['metadata']['article']}]\n{p['text']}"
                           for p in passages)
    skill = Skill(Model(model), prompt=PROMPT, output={"format": {"type": "json_schema", "schema": SCHEMA}})
    result = skill.run(variables={"question": question, "passages": labelled})
    receipt.add(skill.last_usage)
    problems = check_structure(result, SCHEMA)
    if problems:
        print("  not the asked shape:", "; ".join(problems))
    return result


def _norm(text: str) -> str:
    for curly, plain in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'), ("- ", "-")):
        text = text.replace(curly, plain)
    return " ".join(text.split()).casefold()


def _article(label: str) -> str:
    return _norm(re.sub(r"^\s*(article|art\.?)\s*", "", label, flags=re.I))


def verdict(check: dict, result: dict, texts: dict) -> tuple[bool, str]:
    """Whether *result* passes *check*, and why not. *texts*: (regulation, article) → text."""
    citations = result.get("citations") or []
    if check.get("refuse"):
        ok = result.get("refused") is True and not citations
        return ok, "" if ok else "answered a question the documents do not cover"
    want = (_norm(check["regulation"]), _article(check["article"]))
    cited = [c for c in citations if (_norm(c["regulation"]), _article(c["article"])) == want]
    if not cited:
        named = ", ".join(f"{c['regulation']} {c['article']}" for c in citations) or "nothing"
        return False, f"cited {named}, not {check['regulation']} article {check['article']}"
    text = _norm(texts.get((check["regulation"], check["article"]), ""))
    if not any(_norm(c["quote"]) in text for c in cited):
        return False, f"the quote is not in {check['regulation']} article {check['article']}"
    return True, ""
