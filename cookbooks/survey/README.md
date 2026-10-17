# survey — a research brief from a company's filings and the web

Last run: aichain 4.10.1, 2026-10-17 — `--check` 4 of 4: the three figures, and every claim in Results cited.

## The case

An analyst is asked for a one-page brief on a listed company: how it did last
year, what changed, what the press says. Half the answer is in the company's
annual reports; the other half is on the web, and the two do not always agree —
a news story rounds a figure, a blog quotes an old one, a site gets a date
wrong. A research agent that blends both into confident prose hides exactly
the disagreement the analyst needs to see.

This recipe writes the brief on Apple's fiscal 2025 from its last two 10-K
filings, indexed the way `cite` indexes regulations, and from live search. A
lead agent splits the brief into sub-questions and sends each to the member that
can answer it; every claim is cited to a filing Item or a URL, and every figure
on which the web and the filing differ goes into a table instead of being
quietly picked. On the run above, the table caught Investing.com dating the
fourth quarter's end to 30 September; the filing says 27 September.

*Needs: `ANTHROPIC_API_KEY`, `GOOGLE_AI_API_KEY`, one search key (Exa, Tavily,
Parallel, Brave, Perplexity or Firecrawl), `SEC_USER_AGENT`; MarkItDown; a
local Chroma server. First run, measured: $0.05 and about a minute, plus the
embedding calls Google reports no price for.*

## How it works

```
lead ─┬─ delegate → filings: kb.query over both 10-Ks, cites (10-K FY2025, Item 7)
      ├─ delegate → web:     search + fetch("local"), cites the URL
      └─ write_report(markdown, revenue, net_income, employees, …)  →  check("written") ends the run
```

- **The filings are an index, not a prompt.** Two 10-Ks are over half a
  million characters; `index.py` is `cite`'s, split by Item instead of Article,
  so a retrieved passage carries its filing and Item and a citation can name
  them.
- **A team of two, each with one kind of source.** `Agent(..., team=[filings,
  web])` gives the lead a `delegate` tool; each member is an `Agent` with its own
  tools and instructions, and its own conversation. The lead never reads a
  filing or a web page itself.
- **The run ends on a fact.** The lead's last act is `write_report`, a tool that
  writes `data/report.md` and `data/report.json`. `check(lambda s: report
  exists, output=…)` ends the run as a success and returns the figures — the
  report's existence is checked by code, not taken from the model's word.
- **One budget.** `max_cost=0.50` on the lead holds for its members too: a team
  is held to its lead's ceiling.
- **`--check` reads the answer from the filing itself**: revenue and net income
  from the XBRL facts for the fiscal year ending 2025-09-27, the employee count
  from the 10-K's own text (Apple reports none in XBRL). A figure passes within
  0.5 % and with the filing as its source, and every claim in Results — a
  bullet, or a sentence of plain prose — must cite an Item or a URL.

## Running it

```bash
pip install -r requirements.txt chromadb
chroma run --path data/chroma                        # in another terminal
export SEC_USER_AGENT="Jane Doe jane@example.com"    # SEC asks every client to name itself
python run.py --check
```

The search service is the first of Exa, Tavily, Parallel, Brave, Perplexity and
Firecrawl whose key is set. `LEAD_MODEL` (default `claude-sonnet-5-5`) and
`MEMBER_MODEL` (`claude-haiku-5-5`) choose the models. Another company is a new
pair of URLs in `sources.toml` and a new `company` and `period` in
`answers.toml`.

## The original, side by side

[gpt-researcher](https://github.com/assafelovic/gpt-researcher) 0.16.1: a
planner writes research questions, execution agents gather sources for each, a
publisher writes the report.

| | gpt-researcher 0.16.1 | survey |
|---|---|---|
| Your code | about 5 lines | 136 (`team.py` 52, `index.py` 42, `run.py` minus the check) |
| The code behind it | 12,574 lines in the package | aichain's primitives |
| Your own documents | a folder, `DOC_PATH` | an index, cited by filing and Item |
| Web and documents disagreeing | blended into the report | a table of every difference |
| Model vendors in one run | set by configuration | one word per agent |
| Cost | estimated in its README | printed per stage |

Five lines are less to write. The difference is the 136 that say where every
sentence came from and what the sources disagree on — readable top to bottom,
and changed by editing a prompt, not a framework's configuration.
