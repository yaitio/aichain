"""The team: one member reads the filings, one reads the web, a lead writes the brief."""

import json
import os
from pathlib import Path

from yait_aichain import Agent, Model
from yait_aichain.agent import check, step_count
from yait_aichain.tools import fetch, search, tool

DATA = Path(__file__).parent / "data"
LEAD_MODEL = os.getenv("LEAD_MODEL", "claude-sonnet-5-5")
MEMBER_MODEL = os.getenv("MEMBER_MODEL", "claude-haiku-5-5")
#: The search services aichain has, by the key each reads; the first one set is used.
SEARCH = {"exa": "EXA_API_KEY", "tavily": "TAVILY_API_KEY", "parallel": "PARALLEL_API_KEY",
          "brave": "BRAVE_SEARCH_API_KEY", "perplexity": "PERPLEXITY_API_KEY",
          "firecrawl": "FIRECRAWL_API_KEY"}

BRIEF = """Write a one-page brief on {company}'s fiscal year ending {period}: how it did,
what changed from the year before, and what the press says about it.

Split the work into sub-questions and send each to the right member: `filings` for
anything in the annual reports, `web` for coverage. Ask `filings` for revenue, net
income and the number of employees exactly as the 10-K states them.

The brief has four sections:
## Results — bullets, each ending with its citation: a filing Item, as
   (10-K FY2025, Item 7), or a URL.
## Changes — against the year before, cited the same way.
## Coverage — what the press says, each with its URL.
## Disagreements — a table: figure | filing value and Item | web value and URL. List
every figure on which the web and the filing differ; do not pick one. Write "none"
if they agree.

Finish by calling write_report once, with the brief and the three figures from the
filing, each with its source as "10-K FY2025, Item N"."""


@tool(risk="write")
def write_report(markdown: str, revenue: float, revenue_source: str, net_income: float,
                 net_income_source: str, employees: int, employees_source: str) -> dict:
    """Save the finished brief and its three figures.

    Parameters
    ----------
    markdown : str
        The brief, with its four sections.
    revenue : float
        Revenue for the year, in dollars.
    revenue_source : str
        Where it is stated, as "10-K FY2025, Item N".
    net_income : float
        Net income for the year, in dollars.
    net_income_source : str
        Where it is stated.
    employees : int
        The number of employees.
    employees_source : str
        Where it is stated.
    """
    (DATA / "report.md").write_text(markdown)
    figures = {"revenue": {"value": revenue, "source": revenue_source},
               "net_income": {"value": net_income, "source": net_income_source},
               "employees": {"value": employees, "source": employees_source}}
    (DATA / "report.json").write_text(json.dumps({"figures": figures}, indent=2))
    return {"saved": ["data/report.md", "data/report.json"]}


def search_service() -> str | None:
    return next((s for s, key in SEARCH.items() if os.getenv(key)), None)


def build_lead(kb, hooks=None) -> Agent:
    filings = Agent(Model(MEMBER_MODEL), tools=[kb.query], name="filings", hooks=hooks,
                    description="answers from the company's last two 10-K filings, citing the Item",
                    instructions="Answer only from what the filings return. Cite every fact as "
                                 "(10-K FY2025, Item 7) or (10-K FY2024, Item 7). Give figures exactly "
                                 "as the filing states them, with their units.")
    web = Agent(Model(MEMBER_MODEL), tools=[search(search_service()), fetch("local")], name="web",
                hooks=hooks, description="answers from current web sources, citing the URL",
                instructions="Search, read the most relevant pages, and answer with the URL of "
                             "every fact. Give figures exactly as each source states them.")
    return Agent(Model(LEAD_MODEL), team=[filings, web], tools=[write_report], hooks=hooks,
                 max_cost=0.50, name="lead",
                 stop_when=[check(lambda s: (DATA / "report.json").exists(), name="written",
                                  output=lambda s: json.loads((DATA / "report.json").read_text())),
                            step_count(30)])
