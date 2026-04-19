"""
examples/agent_research.py
============================

Agent — two-pass fusion energy research pipeline
=================================================

Demonstrates the improvements added in the latest Agent upgrade:

  ① persona       — Each agent has a specific role injected into every
                    orchestrator system prompt (planning, action, reflection).

  ② FileBackend   — The research agent flushes its memory to disk after the
                    first run.  The writer agent loads those findings as seed
                    variables for the second run — no repeated web fetching.

  ③ History [!]   — The result printer marks important steps (failures,
                    replans, fatal errors) with [!] to mirror the improved
                    _history_summary passed to the orchestrator.

  ④ Focused schema — With multiple tools registered, the action call receives
                    only the schema for the planned tool (Improvement 4).

  ⑤ Middle truncation — MarkItDown returns long page content; the reflection
                        prompt now keeps both the head and the tail of the
                        result instead of silently discarding everything after
                        char 2000.

Two-pass workflow
-----------------

  Phase 1 — Research (waterfall, verbose=1)
      Searches Brave and fetches full article text.
      Flushes the final memory to fusion_research_memory.json.

  Phase 2 — Writing (waterfall, verbose=1, no tools)
      Loads the flushed findings as initial variables.
      Writes a structured analytical report in the requested language.

Setup
-----
    export OPENAI_API_KEY="sk-..."
    export BRAVE_SEARCH_API_KEY="BSA..."
    pip install markitdown

Run
---
    python examples/agent_research.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import Model
from tools  import BraveSearchTool, MarkItDownTool
from agent  import Agent, AgentMemory, FileBackend

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FAST_MODEL    = "gpt-4o-mini"
POWERFUL_MODEL = "gpt-4o"

LANGUAGE      = "English"
MEMORY_FILE   = "fusion_research_memory.json"

# ---------------------------------------------------------------------------
# Models & tools
# ---------------------------------------------------------------------------

orchestrator = Model(POWERFUL_MODEL)
executor     = Model(FAST_MODEL)

search     = BraveSearchTool()
markitdown = MarkItDownTool()

# ---------------------------------------------------------------------------
# Personas  ← Improvement 1
# ---------------------------------------------------------------------------

RESEARCHER_PERSONA = (
    "You are a science journalist with 15 years of experience covering "
    "physics and energy technology.  You specialise in nuclear fusion and "
    "always prioritise concrete data (energy gain ratios, plasma temperatures, "
    "confinement times) over vague descriptions.  When fetching articles, "
    "focus on official lab announcements and peer-reviewed coverage rather "
    "than opinion pieces."
)

WRITER_PERSONA = (
    "You are a senior science editor preparing a briefing for an informed "
    "general audience — readers who understand basic physics but are not "
    "specialists.  Write with authority and precision.  Structure your output "
    "with clear section headings.  Cite specific numbers and source names "
    "wherever the research findings provide them."
)

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

RESEARCH_TASK = """
Search for the most significant nuclear fusion breakthroughs from the past
12 months (2024–2025).  For each major result:

1. Search broadly: "nuclear fusion breakthrough 2024 2025"
2. From the results, identify the 2 most technically significant articles
   (NIF ignition, tokamak records, private-sector milestones, etc.)
3. Fetch the full text of each article using markitdown.
4. Extract and store:
   • The facility / company name
   • What was achieved (energy gain, plasma temperature, confinement time…)
   • The date of the result
   • A 2–3 sentence technical summary

Store everything clearly in memory so it can be used to write a report.
""".strip()

REPORT_TASK = """
Using the research findings already in your memory variables, write a
comprehensive analytical report on nuclear fusion energy in {language}.

Structure:
  1. Executive Summary (3–4 sentences)
  2. Key Technical Breakthroughs — one subsection per development,
     with specific numbers and the source name
  3. Competing Approaches — tokamak vs. inertial confinement vs. private
     ventures (cite specific companies from the findings)
  4. Remaining Engineering Challenges
  5. Realistic Timeline for commercial power
  6. Your Expert Assessment — which bets look most promising and why

Be specific, analytical, and cite every source by name.
The report should be suitable for publication.
""".strip()


# ---------------------------------------------------------------------------
# Result printer  ← marks [!] important steps (Improvement 2)
# ---------------------------------------------------------------------------

def print_result(result, label: str) -> None:
    bar = "─" * 70
    print(f"\n{bar}")
    print(f"  {label}")
    print(bar)

    if not result:
        print(f"\n✗  Failed: {result.error}")
        print()
        return

    print(
        f"\n✓  Success — {result.steps_taken} step(s) · "
        f"{result.tokens_used:,} tokens\n"
    )
    print("Execution trace  ([!] = important step: failure / replan / fatal):")
    for rec in result.history:
        assessment = rec["reflection"].get("assessment", "?")
        decision   = rec["reflection"].get("decision", "?")
        stored     = rec.get("stored_as") or "—"
        goal       = rec["step_goal"][:65]
        tokens     = rec["tokens"]
        err        = rec.get("exec_error")

        # Mirror the _is_important logic used by the improved _history_summary
        important = (
            err is not None
            or assessment in {"failure", "fatal", "partial"}
            or decision   in {"replan", "stop", "final_answer"}
        )
        flag = "[!]" if important else "   "
        err_note = f"\n      ⚠  {str(err)[:90]}" if err else ""

        print(
            f"  {flag} Step {rec['step']+1}/{rec['attempt']}"
            f" [{rec['action_type']}] {goal!r}\n"
            f"       → {assessment}  ·  {decision}  ·  "
            f"stored='{stored}'  ·  {tokens:,} tok"
            f"{err_note}"
        )

    if result.memory:
        print(f"\n  Memory keys: {sorted(result.memory.keys())}")
    print()


# ---------------------------------------------------------------------------
# Phase 1 — Research  (Improvements 1, 4, 5)
# ---------------------------------------------------------------------------

def run_research_phase() -> dict:
    print("\n" + "═" * 70)
    print("  Phase 1 — Research")
    print("  Persona : researcher / science journalist")
    print(f"  Backend : FileBackend({MEMORY_FILE!r})")
    print("═" * 70)

    research_memory = AgentMemory(
        backend=FileBackend(MEMORY_FILE)   # ← Improvement 5
    )

    agent = Agent(
        orchestrator = orchestrator,
        executors    = [executor],
        tools        = [search, markitdown],   # two tools → focused schema per step
        mode         = "waterfall",
        max_steps    = 6,
        max_attempts = 2,
        max_tokens   = 60_000,
        memory       = research_memory,
        verbose      = 1,
        name         = "fusion_researcher",
        persona      = RESEARCHER_PERSONA,     # ← Improvement 1
    )

    print(f"\nAgent: {agent!r}\n")

    result = agent.run(task=RESEARCH_TASK)
    print_result(result, "Phase 1 — Research Complete")

    # Flush findings to disk so Phase 2 can load them  ← Improvement 5
    research_memory.flush()
    print(f"  → Flushed {len(result.memory)} memory key(s) to {MEMORY_FILE!r}")
    print(f"  → Keys: {sorted(result.memory.keys())}\n")

    return result.memory   # plain dict snapshot


# ---------------------------------------------------------------------------
# Phase 2 — Report writing  (Improvements 1, 5)
# ---------------------------------------------------------------------------

def run_writing_phase(findings: dict) -> None:
    print("═" * 70)
    print("  Phase 2 — Writing")
    print("  Persona : senior science editor")
    print(f"  Input   : {len(findings)} key(s) loaded from research phase")
    print("═" * 70)

    # Load persisted findings — demonstrates that flush() actually wrote data
    saved = FileBackend(MEMORY_FILE).load()   # ← Improvement 5
    print(f"\n  Verified: {len(saved)} key(s) reloaded from {MEMORY_FILE!r}")
    print(f"  Keys    : {sorted(saved.keys())}\n")

    agent = Agent(
        orchestrator = orchestrator,
        executors    = [executor],
        tools        = [],               # no tools — all data is in variables
        mode         = "waterfall",
        max_steps    = 4,
        max_attempts = 2,
        max_tokens   = 40_000,
        verbose      = 1,
        name         = "fusion_writer",
        persona      = WRITER_PERSONA,   # ← Improvement 1, different persona
    )

    print(f"Agent: {agent!r}\n")

    result = agent.run(
        task      = REPORT_TASK,
        variables = {**findings, "language": LANGUAGE},  # inject research findings
    )
    print_result(result, f"Phase 2 — Final Report ({LANGUAGE})")

    if result:
        print("─" * 70)
        print(f"  Report:")
        print("─" * 70)
        print(result.output)
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n  Agent — Two-Pass Fusion Energy Research Pipeline")
    print(f"  Models : orchestrator={POWERFUL_MODEL}, executor={FAST_MODEL}")
    print(f"  Tools  : {[search.name, markitdown.name]}")
    print(f"  New    : persona · FileBackend · focused schema · mid-truncation")

    findings = run_research_phase()
    run_writing_phase(findings)

    # Clean up temp file
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)
        print(f"  Cleaned up {MEMORY_FILE!r}")
