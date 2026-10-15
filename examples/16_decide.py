"""
16_decide.py — A closed set of answers with a visible price: classify support messages.

Keys: OPENAI_API_KEY.
"""

from yait_aichain.tools import decide

CATEGORIES = ["billing", "bug", "feature request", "account access", "other"]
MESSAGES = ["I was charged twice for October.",
            "The export button does nothing when I click it.",
            "Could you add a dark mode?"]

triage = decide("openai", questions={
    "category": {"type": "choice", "choices": dict.fromkeys(CATEGORIES),
                 "instructions": "Which kind of support request is this?"}})

for text in MESSAGES:
    answer = triage.run(text)
    category = answer["answers"]["category"]
    print(f"{category['choice']:<16} {category['confidence']:.2f}  ${answer['cost']:.6f}  {text}")
