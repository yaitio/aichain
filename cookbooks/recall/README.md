# recall — an assistant that remembers you, without a vector database

Last run: aichain 4.10.1, byheart 0.1.1, 2026-10-17, `claude-haiku-5-5`, conversation 0, three runs — F1 0.562, 0.573, 0.574 (threshold 0.5, **met**); refusals 0.872 each time (threshold 0.9, **not met**). See [What the numbers say](#what-the-numbers-say).

## The case

A personal assistant is only useful if it remembers: the person mentioned last
month that they are training for a marathon and that their daughter starts
school in September; asked today *what should I plan for the autumn?*, a good
answer uses both. The common answer is a memory service that extracts "facts"
from every exchange with extra model calls and stores them in a vector
database. byheart keeps the conversation itself and selects from it, with no
model call, no embeddings and no server.

This recipe replays a real long conversation from a public benchmark, LoCoMo —
419 turns over 19 sessions and five months between two friends — into byheart,
then asks the benchmark's own 199 questions in a fresh process and scores the
answers against the benchmark's answers, with the benchmark's own metric.

LoCoMo is licensed CC BY-NC 4.0: non-commercial use only. The recipe downloads
it when it runs and redistributes nothing.

*Needs: `ANTHROPIC_API_KEY`; `yait-byheart`, `nltk`. No services. A run,
measured: $0.17 and about half a minute.*

## How it works

```
LoCoMo session by session ─ Memory.append(role, "Name: text", when=date) ─ data/memory.json   0 calls
                                                                                │  a new process
question ─ Memory.compose(question) ─ Skill(input={"messages": [system, *messages]}) ─ answer   1 call
```

- **Ingestion is appending.** Each turn goes into `Memory(budget=20_000)` as it
  was said, with its session's date; after every session the memory is saved
  with `to_dict()` to `data/memory.json`. It is plain JSON: you can read it.
  The receipt's `ingest` row is zero calls, zero tokens.
- **A fresh process answers.** `run.py` hands over to `answer.py` in a new
  process, which knows only the JSON file. `Memory.from_dict()` restores it,
  and for every question `compose()` returns the messages to send — the whole
  conversation while it fits the budget, otherwise the turns that matter for
  this question, dated — and a `Skill` sends them.
- **Scored the benchmark's way.** `answer.py` reproduces LoCoMo's
  `task_eval/evaluation.py`: token F1 after its normalisation and Porter
  stemming, multi-hop answers matched part by part, and the 47 adversarial
  questions — which ask about something the conversation does not say —
  counted as right only when the answer is "not mentioned" or "no information
  available". `data/answers.json` keeps every answer with its score.

### What the numbers say

The answering prompt was written against conversation 1 — F1 0.554, refusals
0.958 there — and conversation 0 was run only after, so the numbers above are
on questions the prompt was not tuned to.

On conversation 0 the model answered 6 of the 47 adversarial questions instead
of declining them. Two of the six are right and the metric cannot read them:
asked *Did Caroline make the black and white bowl?* it answered *No, Melanie
made it*, and *Is Oscar Melanie's pet?* got *No, Oscar is Caroline's guinea
pig* — both correct corrections of a false premise, and both scored 0 because
the metric looks for the words "not mentioned". The other four are real
mistakes: an event or an object given to the wrong friend. The threshold was
named in advance and is not met; the metric is LoCoMo's and is not changed
after the fact.

**byheart's published 87.3 % is a different number.** It is a J-score — an LLM
judge, Mem0's — on all ten conversations. This recipe's F1 is LoCoMo's own
string metric on one. The two are not comparable, and this page does not
compare them.

## Running it

```bash
pip install -r requirements.txt
python run.py --check                  # conversation 0
python run.py --conversation 1 --check
python run.py --model claude-sonnet-5-5
```

## The original, side by side

Mem0's quickstart:

```python
from mem0 import Memory
m = Memory()
m.add(messages, user_id="alice")
m.search("what should I plan for the autumn?", user_id="alice")
```

| | Mem0 | recall |
|---|---|---|
| Model calls per ingested exchange | an LLM call to extract facts, and embeddings | none |
| What is stored | facts a model extracted | the conversation, verbatim |
| Services | a vector store and an embedding model | none |
| What an answer is built from | the facts retrieved | the turns selected, with their dates |
| Lines of the recipe | — | `ingest.py` 23, `answer.py` 80 with the scoring |

Four lines are less to write than these. What they buy is a model call on
every message a person ever sends, and a store of what a model decided was
worth keeping instead of what was said.
