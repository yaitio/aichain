"""Answer the benchmark's questions in a fresh process, from data/memory.json alone,
and score them the way LoCoMo does.

    python answer.py SAMPLE.json [--model M] [--check]
"""

import argparse
import json
import re
import string
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from nltk.stem import PorterStemmer
from yait_byheart import Memory

from yait_aichain import Model, Skill

from receipt import Receipt

DATA = Path(__file__).parent / "data"
MODEL = "claude-haiku-5-5"
SYSTEM = ("You remember a long conversation between two friends, {a} and {b}. Answer the question "
          "with the shortest phrase that answers it — a name, a date, a few words — with no "
          "sentence around it and no explanation. Give a date the way the conversation dates it: "
          "7 May 2023, or the year alone if that is all it says. Mind who did what: if the "
          "question gives one person what the other did, or the conversation does not say, "
          "answer exactly: not mentioned.")
ADVERSARIAL = 5          # LoCoMo's category for questions the conversation does not answer
_stem = PorterStemmer().stem


def _normal(text: str) -> list[str]:
    """LoCoMo's normalisation: no commas, punctuation, articles or "and"; stemmed."""
    text = "".join(c for c in str(text).replace(",", "").lower() if c not in string.punctuation)
    return [_stem(w) for w in re.sub(r"\b(a|an|the|and)\b", " ", text).split()]


def _f1(prediction: str, truth: str) -> float:
    p, t = _normal(prediction), _normal(truth)
    same = sum((Counter(p) & Counter(t)).values())
    return 0.0 if not same else 2 * same / (len(p) + len(t))


def score(qa: dict, prediction: str) -> float:
    """LoCoMo's task_eval/evaluation.py, category by category."""
    if qa["category"] == ADVERSARIAL:
        return float(any(s in prediction.lower() for s in ("not mentioned", "no information available")))
    truth = str(qa["answer"])
    if qa["category"] == 3:
        truth = truth.split(";")[0].strip()
    if qa["category"] == 1:                    # multi-hop: every part of the answer, best match each
        parts = [p.strip() for p in prediction.split(",")]
        return sum(max(_f1(p, t.strip()) for p in parts) for t in truth.split(",")) / len(truth.split(","))
    return _f1(prediction, truth)


def answer(memory: Memory, question: str, model: str, system: str):
    messages = memory.compose(question)["messages"]
    skill = Skill(Model(model), input={"messages": [{"role": "system", "parts": [system]}, *messages]})
    try:
        return skill.run(), skill.last_usage
    except Exception as exc:
        return f"(error: {type(exc).__name__})", skill.last_usage


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_argument("sample")
    cli.add_argument("--model", default=MODEL)
    cli.add_argument("--check", action="store_true")
    args = cli.parse_args()
    memory = Memory.from_dict(json.loads((DATA / "memory.json").read_text()))
    sample = json.loads(Path(args.sample).read_text())
    qas, people = sample["qa"], sample["conversation"]
    system = SYSTEM.format(a=people["speaker_a"], b=people["speaker_b"])

    receipt = Receipt()
    with ThreadPoolExecutor(8) as threads:
        done = list(threads.map(lambda qa: answer(memory, qa["question"], args.model, system), qas))
    with receipt.stage("ingest"):
        pass                                   # byheart keeps the conversation without a model call
    with receipt.stage("answer"):
        for _, usage in done:
            if usage is not None:
                receipt.add(usage)

    f1s, refusals, record = [], [], []
    for qa, (prediction, _) in zip(qas, done):
        (refusals if qa["category"] == ADVERSARIAL else f1s).append(score(qa, prediction))
        record.append({**qa, "prediction": prediction, "score": score(qa, prediction)})
    (DATA / "answers.json").write_text(json.dumps(record, indent=1))
    for qa, (prediction, _) in list(zip(qas, done))[:5]:
        print(f"Q: {qa['question']}\n   → {prediction}   (key: {qa.get('answer', 'not mentioned')})")
    receipt.print()
    mean_f1, refused = sum(f1s) / len(f1s), sum(refusals) / len(refusals)
    print(f"\nF1 {mean_f1:.3f} over {len(f1s)} questions (threshold 0.5)\n"
          f"refusals {refused:.3f} over {len(refusals)} adversarial questions (threshold 0.9)")
    if args.check:
        sys.exit(0 if mean_f1 >= 0.5 and refused >= 0.9 else 1)


if __name__ == "__main__":
    main()
