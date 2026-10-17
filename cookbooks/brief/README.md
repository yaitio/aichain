# brief — a report longer than any single answer

Last run: aichain 4.10.1, 2026-10-17 — `--check` 3 of 3, three runs out of three: 9,633 to 10,718 words.

## The case

A team lead asks for a literature brief: twelve recent papers on one topic,
summarised and compared, ten pages. No model writes ten pages in one answer —
the output limit cuts it off, sometimes mid-sentence, sometimes silently. The
usual workaround is to ask again and paste the halves together, and the result
repeats itself and loses the thread.

This recipe treats the limit as the shape of the program: a cheap model
summarises each paper in parallel, a strong model plans the brief and then
writes it one section at a time, each section seeing the summaries of its
papers and the sections already written. No call produces the whole document,
and none is cut off.

The topic is how to measure whether an LLM agent is reliable: twelve arXiv
papers from τ-bench to judge-reliability harnesses, every one CC BY 4.0.

*Needs: `ANTHROPIC_API_KEY`; MarkItDown. No services. First run, measured:
about $0.60 and six minutes; $0.45 after, when the summaries are kept.*

## How it works

```
12 PDFs ─ convert ─ Pool(Skill(haiku), max_flows=6) ─ 12 summaries  ─┐
                                                                     ├─ Skill(sonnet, JSON schema) ─ outline: 6–10 sections
                                                                     └─ for each section: Skill(sonnet) with the outline,
                                                                          its papers' summaries, everything written so far
```

- **Summaries in parallel, by a cheap model.** `Pool` runs one `Skill` over
  twelve papers six at a time; each summary is 300 words under four fixed
  headings — question, method, result, limits — so the writer gets the same
  shape twelve times. They are kept in `data/summaries.json`.
- **The outline is data.** One call returns the sections as JSON — a title and
  the papers each draws on — so the loop below is ordinary Python, and every
  paper is placed before a word is written.
- **One section per call, about 1,100 words.** `write.py` is a thirty-line loop.
  Each call has room for one section and no more (`max_tokens=3000`), sees what
  is already written so it does not repeat it, and runs with
  `reasoning="none"`: thinking tokens count against the same ceiling and would
  cut the section short. A section that still comes back cut —
  `skill.last_truncated` — is asked again once for half the length, and a
  second cut fails the run rather than keep half a sentence.
- **`--check`** asks three things of the result: that it is longer, in output
  tokens, than the writer can give in one answer (8,192 by default); that no
  section was cut; and that each of the twelve papers is cited at least once.

### The one-call version, beside it

```bash
python run.py --one-call
```

asks the same model for the whole brief at once, about 9,000 words. On the run
above it returned 2,826 words and stopped mid-sentence — *"…which produces a
claim of the form "reliability is at"* — with `last_truncated` set and a warning
that the answer was cut. This recipe's brief is three to four times as long and
every section ends where it means to.

## Running it

```bash
pip install -r requirements.txt
python run.py --check
python run.py --one-call
```

`WRITER_MODEL` changes the writer (default `claude-sonnet-5-5`); the summaries
are `claude-haiku-5-5`'s. Another topic is another twelve entries in
`sources.toml` and `TOPIC` in `run.py`. Delete `data/summaries.json` to
summarise again.
