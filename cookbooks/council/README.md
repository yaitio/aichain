# council

Last run: aichain 4.10.1, 2026-10-17 — all four scripts, with `gpt-6.1-sol` out of the panel for want of credit.

Ask several models the same question and see whether they agree — and measure
whether their agreement means they are right.

*Seven models from seven providers by default — see [Running it](#running-it).
`truth.py` on 100 questions: $0.26 and about twenty minutes.*

## The case

A model states an invented fact in exactly the same confident register as a
true one. There is no tell in the text, so the question is what cheap signal
you can get about whether an answer is worth trusting.

Asking the same model again is a weak signal. **Repetition measures sampling
temperature, not truth**: if a model invented something on the first pass it
will usually invent the same thing on the fifth, because the invention is where
its weights point.

Asking *different* models measures something else. Different weights, different
training data, different failure modes — to agree on a fabrication they have to
fabricate the **same** thing, which is a much narrower coincidence. What comes
out is agreement or disagreement, not a better answer. Agreement lowers the
probability of a fabrication; it does not remove it, because these models share
training data and a widely repeated error is one they will repeat together.

Turning several answers into one verdict has more than one solution, and they
do not behave alike — which matters here, because a mechanism that suppresses
disagreement is the wrong instrument for detecting one. Three are here, because
at forty lines apiece you can compare them instead of committing to the first:

| script | how agreement is reached |
|---|---|
| [`council.py`](council.py) | everyone answers, everyone ranks the answers **blind**, a chairman writes the final one |
| [`parallel.py`](parallel.py) | the same, with the two fan-out stages run at once |
| [`consensus.py`](consensus.py) | no chairman and no vote: models take turns until a full lap passes with nobody wanting to change anything |
| [`truth.py`](truth.py) | a vote by letter on questions with a known answer: is the agreement evidence of truth? |

The panel has to be models from **different vendors**, which is the only reason
any of this costs code at all. The default roster is seven: four from the
western labs and three from Chinese ones, each addressed directly.

**They are not interchangeable, and the choice between them is not about
speed.** Run the same question through the vote and through the conversation
with the same seven models and you get two different pictures: the vote spread
them from an average rank of 1.71 to a flat 7.00 — one model placed last by
every one of the other six — while the conversation reached unanimity on the
first proposal, in eight calls instead of fifteen. **The conversation is not
measuring agreement; it is measuring willingness to defer.** For deciding
whether to believe an answer, the noisy mechanism is the useful one.

**Is agreement evidence of truth? Measured.** TruthfulQA asks questions built
to catch common misconceptions, with one true option among several. `truth.py`
puts 100 of them to a panel — Claude Haiku 5.5, Gemini 3.8 Flash, Grok 4.3,
DeepSeek Flash, Qwen 3.7 Plus and Mistral Small, 2026-10-17 — and scores the
majority's letter against the key:

```
              right  wrong
unanimous        69      0
majority         23      4
split             —      4
```

Unanimity was never wrong, a majority was right 23 times in 27, and a split
vote had no answer to give. So agreement is evidence. It is not a better
answer: the panel got 92 right, and its best member alone got 99 — on seven
of the eight questions the panel got wrong, that member had the right letter
and the vote lost it. A council is a way
to know how much to trust an answer, not a way to make one.

On 20 questions every member was right every time, and the table had nothing
in its wrong column; a sample at its ceiling cannot say whether agreement means
anything, which is why the default is 100.

## How it works

### council.py

Three stages. Everyone answers; the answers are relabelled `Response A`, `B`,
`C` so a judge cannot recognise a house style, its own included; everyone ranks
the relabelled set; the chairman gets the answers plus the standings and writes
the result.

```python
answers = [ask(m, QUESTION) for m in COUNCIL]
reviews = [ask(m, RUBRIC, question=QUESTION, answers=block) for m in COUNCIL]
```

**Prompts are templates and the library fills them.** Each prompt is a constant
with `{slots}`; `ask()` hands the values to `Skill.run()`. The library replaces
only bare `{identifier}` placeholders that have a matching key — unknown
placeholders, format specs and literal braces such as a JSON example are left
alone and never raise. `str.format`, which this recipe used first, turns any
brace in a model's answer into a crash.

### parallel.py

Members answer independently, so nothing makes them wait. Three lines of
`concurrent.futures` and the two comprehensions run at once.

**Not `Pool`.** `Pool` runs **one runner over many inputs**; this is **one
input over many models**. A `Skill` binds its model at construction, `run()`
takes no override, and a list of models is a *fallback chain* for transient
failures, not a fan-out. Forcing it to fit needs a `Tool` whose `model`
argument is not a tool argument at all — it is the runner smuggled through the
item dict. Ten lines to misuse a primitive where three lines of the standard
library say what is happening.

### consensus.py

No aggregator. Each model in turn sees the discussion and either replies
`AGREE` or proposes a replacement; consensus is one full lap of nothing but
`AGREE`. `MAX_LAPS` is a ceiling, not a plan — two stubborn members can trade
proposals forever.

It is the cheapest of the three and the quickest to agree, and both of those
are reasons to distrust it as a hallucination check. Widening the panel from
four models to seven, across four different labs, made the *vote* far more
divisive and left the *conversation* converging faster than before — first
proposal accepted, one lap, done. Whatever it is that a first speaker's
proposal has going for it, being right is not obviously part of it.

**Each model sees the conversation as it lived it.** The transcript is rebuilt
per speaker: its own earlier turns are `assistant`, everyone else's are `user`
naming who spoke.

```python
turns.append({"role": "assistant", "parts": [what]} if who == speaker else
             {"role": "user",      "parts": [f"{who} says:\n{what}"]})
```

Flattening the discussion into one block of text instead makes models pile onto
each other's answers rather than converge on one.

**An `AGREE` is a turn like any other.** Recording only the rewrites hides half
the conversation — a late speaker cannot see that the others have already
settled.

**The protocol lives in the system message,** not in a reminder appended to
every call: the previous speaker's turn is the last thing in the array, and a
conversation does not restate its own rules on every line. That only works
because every turn is recorded — a trailing `assistant` message is a prefill,
and providers reject it.

### truth.py

Each member answers through a `Skill` with a JSON schema whose only field is an
`enum` of the question's letters, so an answer cannot be anything but a letter
of this question. The sample is seeded, the options are shuffled with the same
seed, and the set is downloaded once and checked by hash (`sources.toml`).

Every script prints a receipt, one row per model.

## Running it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   OPENAI_API_KEY=...     GOOGLE_AI_API_KEY=...
export XAI_API_KEY=...         DEEPSEEK_API_KEY=...   MOONSHOT_API_KEY=...
export DASHSCOPE_API_KEY=...   MISTRAL_API_KEY=...

python council.py              # vote and chairman
python parallel.py             # the same, in parallel
python consensus.py            # dialogue until agreement
python truth.py                # 100 questions with a known answer
python truth.py --questions 20
```

`GOOGLE_AI_API_KEY`, not `GEMINI_API_KEY` — that is the variable the provider
table reads.

**Fewer keys is fine.** A member without a key sits out, and one that fails —
no credit, an outage — drops out of the panel while the rest carry on, in all
four scripts; `truth.py` needs three. Edit `COUNCIL`, `CHAIRMAN` or `MEMBERS`
at the top of each file to choose your own.

**The vote does not scale linearly.** Its second stage has every member read
every answer, so seven members is roughly four minutes and fifteen calls where
four was two minutes and nine. The conversation grows far more gently — eight
calls for seven members — for the reason given above, which is not a
recommendation.

Change `QUESTION` to your own — but the mechanism is only interesting where
there is a fact to get wrong. On a matter of taste there is nothing to
fabricate and nothing for the panel to catch.

## The original, side by side

[karpathy/llm-council](https://github.com/karpathy/llm-council/tree/92e1fccb1bdcf1bab7221aa9ed90f9dc72529131)
at `92e1fcc`: members answer, rank each other anonymously, a chairman writes
the result — the shape of `council.py`.

| | llm-council | council.py |
|---|---|---|
| Lines of orchestration | 332 (`council.py` 251, `openrouter.py` 62, `config.py` 19) | 60 |
| Reaching the models | through OpenRouter | each vendor directly, one word per model |
| A member that fails | fails the stage | drops out, the rest go on |
| Cost of a run | not reported | per model |
| Web server, storage, React frontend | its own | not needed |

Beside it, `truth.py` asks the question the original leaves open: whether the
agreement it computes is worth anything.
