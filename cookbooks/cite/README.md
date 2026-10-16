# cite — answers from your documents, with the article that says so

Last run: aichain 4.9.0, 2026-10-17 — `--check` 13 of 13, three runs out of three.

## The case

A compliance officer has three EU regulations open — the GDPR, the AI Act and
the Data Act, about 400 pages together — and a question from the product team:
*what is the maximum fine if our model breaks the AI Act's transparency rules,
and is it the same as under the GDPR?* The answer exists, in two articles of two
documents. What they need is not a paragraph that sounds right but the article
to quote in their reply — and an honest "these documents do not say" when the
question is about something else.

That is the most requested shape in applied AI, and every framework has a
quickstart for it. This recipe walks the whole path a fork has to walk:
download, convert, read a scanned page, split by article, chunk, embed, store,
retrieve, rerank, answer with a citation, and check that the quote is really
in the article it names. The document converter comes from outside, as a tool.
Each step's provider is one word, and the receipt says what indexing and every
answer cost.

The scanned page is page 7 of the US Privacy Act of 1974 as printed in 1974: a
picture with no text layer, so one question in the answer key can be answered
only by reading it.

*Needs: `ANTHROPIC_API_KEY`, `GOOGLE_AI_API_KEY`, and `COHERE_API_KEY` for the
rerank (or `--no-rerank`); MarkItDown; a local Chroma server. First run,
measured without the rerank: $0.094 and 104 seconds, plus 14 embedding calls
Google reports no price for.*

## How it works

```
sources.toml ─ fetch_sources ─ convert (MarkItDown, a @tool) ─ split by article ─ chunk ─┐
                                   └ a page with no text: a vision Skill             │
                                                                                     ▼
question ─ kb.query (20) ─ rerank (6) ─ Skill with a JSON schema ─ answer + citations ─ check
```

- **`convert.py` is MarkItDown as a tool.** aichain does not convert
  documents and does not mean to: a converter is a heavy dependency with its
  own release cycle. `@tool(risk="read")` makes it an aichain tool — callable
  from code, as here, or by an agent — in thirty lines. A page MarkItDown
  returns no text for is read by a vision `Skill`. That is Gemini, not Claude:
  every Claude model answered a whole-page transcription with "Output blocked
  by content filtering policy".
- **The split is by article, before chunking.** A chunk then carries its
  regulation and article as metadata, so a citation names an article rather
  than a file and a page. The regex is one constant; the annexes are cut off
  first, because they quote article numbers as headings.
- **`Vectors("chroma", …)` holds the index**, embedded by
  `embed("google", model="gemini-embedding-001")` — Anthropic has no
  embeddings. A second run finds the store full and makes no embedding call.
- **The answer is a `Skill` with a JSON schema**: `refused`, `answer`,
  `citations` with a verbatim `quote`. The schema makes the refusal a field,
  not a tone of voice, and makes the citations something code can check.
- **`--check` checks the quotes**, not the prose: a citation passes when it
  names the right article *and* its quote is a substring of that article's
  converted text. A plausible paragraph with the wrong article fails. The
  AI Act puts the same EUR 15 000 000 in Article 99 and in Article 101; only
  one is about transparency.

## Running it

```bash
pip install -r requirements.txt chromadb
chroma run --path data/chroma            # in another terminal
python run.py --check                    # the thirteen questions, scored
python run.py "Can a regulator fine a public authority under the GDPR?"
```

| Variable | Default | |
|---|---|---|
| `ANSWER_MODEL` | `claude-sonnet-5-5` | or `--model`; any model aichain knows |
| `VISION_MODEL` | `gemini-3.8-flash` | reads the scanned page |
| `CHROMA_URL` | `http://localhost:8000` | |

Without `COHERE_API_KEY`, `--no-rerank` takes the store's top six instead of
reranking twenty; the thirteen checks pass either way on this set. `--refresh`
downloads, converts and indexes again. To change the embedder, change one line
of `index.py`, and the dimension with it:

```diff
-EMBEDDER, DIMENSION = ("google", "gemini-embedding-001"), 1536
+EMBEDDER, DIMENSION = ("openai", "text-embedding-3-small"), 1536
```

## The original, side by side

LlamaIndex 0.14.25, the starter tutorial:

```python
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What is the maximum GDPR fine?")
```

| | LlamaIndex starter | cite |
|---|---|---|
| Lines of orchestration | 4 | 111 (`index.py` 53, `answer.py` 58) |
| A citation names | a file and a node | a regulation and an article, with a verbatim quote |
| A question the documents do not answer | an answer in the same voice | `refused: true`, checked |
| A scanned page | not read | read by a vision model |
| The index between runs | rebuilt (persisting: three more lines) | kept in the store |
| Checked against a key | — | 13 questions, quotes verified |
| Cost of a run | not reported | per stage |
| Vector server, converter | needed by both in production | needed by both |

Four lines answer a question; they do not say which article answers it, or
notice when none does. The difference is the 107 lines that do — readable top
to bottom, with no framework class between you and the request.
