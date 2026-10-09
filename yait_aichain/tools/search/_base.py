"""
tools.search._base
==================

``Search`` — one format of options and results for every web search provider.

The providers sell the same thing — ranked pages for a query, filtered by
time, place and site — and spell every part of it differently: Brave filters
time with ``freshness``, Firecrawl with Google's ``tbs``, Perplexity with
``search_recency_filter`` and MM/DD/YYYY dates, Exa with ISO dates; two of the
four have no site filter and take ``site:`` in the query. This class holds the
meaning; each provider translates it.

Options — one vocabulary
------------------------
``max_results``  how many results; kept inside the provider's ceiling.
``recency``      ``hour`` | ``day`` | ``week`` | ``month`` | ``year``.
``start_date``   only pages published on or after this ``YYYY-MM-DD``.
``end_date``     only pages published on or before this ``YYYY-MM-DD``.
``country``      ISO 3166-1 alpha-2, e.g. ``"GB"``.
``language``     ISO 639-1, e.g. ``"en"``.
``domains``      only these sites — a domain, or a domain with a path prefix.

``recency`` and ``start_date`` both bound the same side; the later of the two
is the one both promises keep. A provider adds its own options beside these
(``extra_options``), and keeps them.

An option a provider cannot honour is never dropped in silence: it is
reported — as a warning once per provider and option, as a line at the end of
the text the model reads, and in the result's ``notices``. An
option it honours differently (``hour`` on a provider whose finest window is a
day) is adapted and reported the same way.

Results — one record
--------------------
``run()`` returns ``{"query", "provider", "results", "notices", "cost",
"cost_basis", "usage"}`` as a :class:`~tools.Record`. ``cost`` is the call's price in USD; ``cost_basis``
says where it came from — the provider's own figure, the provider's units
(credits, operation counts) at the published rate, or the published rate alone
for a provider that reports nothing — and ``usage`` holds those units as the
provider sent them. Rates are each tool's, with their source and date. Each
result is::

    {"title": str, "url": str, "date": "YYYY-MM-DD" or "", "author": str,
     "snippets": [str, ...], "text": str}

``date`` is the publication date as the provider reports it, normalised to a
calendar day, or empty — not a fact about the page: providers date pages
wrongly, and a caller that needs the true date reads it from the page
(``fetch``). ``text`` is the page's own text when the provider returned it
(Exa, on request), else empty.

The record is one object for three readers: code reads its fields, an agent
shows the model :meth:`Search.render` — the results as a numbered list with
the notes at the end — and a Skill prompt that substitutes it gets the same
list. ``search()`` is ``run()`` under its 3.2 name.

``search(service)`` makes one: ``search("exa")``, ``search("perplexity")``.
"""

from __future__ import annotations

import os
import re
import warnings
from datetime import date, datetime, timedelta, timezone

import urllib3

from ...clients._base import make_http
from .._base import Record, Tool

# ---------------------------------------------------------------------------
# Shared network policy for all search providers
# ---------------------------------------------------------------------------
# Search endpoints answer in seconds; without a timeout a hung server blocks
# the calling pipeline (or agent) forever.  Retries cover transient 429/5xx;
# search POSTs are read-only queries, so retrying them cannot double-bill a
# generation the way a retried completion call could.
SEARCH_TIMEOUT = urllib3.Timeout(connect=10.0, read=120.0)
SEARCH_RETRIES = urllib3.Retry(
    total=3,
    backoff_factor=1.0,
    status_forcelist={429, 500, 502, 503, 504},
    allowed_methods=None,   # include POST — see note above
    raise_on_status=False,
)

RECENCY = {"hour": timedelta(hours=1), "day": timedelta(days=1), "week": timedelta(weeks=1),
           "month": timedelta(days=31), "year": timedelta(days=366)}

STANDARD_OPTIONS = {
    "max_results": {"type": "integer", "description": "How many results to return (default 10)."},
    "recency":     {"type": "string", "enum": list(RECENCY),
                    "description": "Only pages published within this window."},
    "start_date":  {"type": "string", "description": "Only pages published on or after this date, YYYY-MM-DD."},
    "end_date":    {"type": "string", "description": "Only pages published on or before this date, YYYY-MM-DD."},
    "country":     {"type": "string", "description": "ISO 3166-1 alpha-2 country code, e.g. 'GB'."},
    "language":    {"type": "string", "description": "ISO 639-1 language code, e.g. 'en'."},
    "domains":     {"type": "array", "items": {"type": "string"},
                    "description": "Only these sites — a domain, or a domain with a path prefix."},
}

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def day(value) -> str:
    """A provider's date as ``YYYY-MM-DD``, or ``""`` when it is not one.

    Takes ISO dates and timestamps, ``Sep 30, 2026``, ``30 Sep 2026`` and the
    relative ``3 days ago`` that Google answers with for recent pages.
    """
    if not value:
        return ""
    # An RFC 1123 weekday prefix — "Tue, 11 Mar 2025 17:00:00 GMT" (Tavily) — says nothing more.
    s = re.sub(r"^[A-Za-z]{3},\s*", "", str(value).strip())
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return "-".join(m.groups())
    m = re.match(r"([A-Za-z]{3})[a-z]*\.? (\d{1,2}),? (\d{4})", s) or None
    if m and m.group(1).lower() in _MONTHS:
        return f"{m.group(3)}-{_MONTHS[m.group(1).lower()]:02d}-{int(m.group(2)):02d}"
    m = re.match(r"(\d{1,2}) ([A-Za-z]{3})[a-z]*\.? (\d{4})", s)
    if m and m.group(2).lower() in _MONTHS:
        return f"{m.group(3)}-{_MONTHS[m.group(2).lower()]:02d}-{int(m.group(1)):02d}"
    m = re.match(r"(\d+) (minute|hour|day|week|month)s? ago", s)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        span = {"minute": timedelta(minutes=n), "hour": timedelta(hours=n), "day": timedelta(days=n),
                "week": timedelta(weeks=n), "month": timedelta(days=30 * n)}[unit]
        return (datetime.now(timezone.utc) - span).date().isoformat()
    return ""


def _check_day(name: str, value) -> str:
    try:
        return date.fromisoformat(str(value)[:10]).isoformat()
    except ValueError:
        raise ValueError(f"{name} must be a date, YYYY-MM-DD; got {value!r}") from None


class Search(Tool):
    """
    Base of the web search tools: one vocabulary of options, one record of
    results, a provider per subclass.

    A subclass sets ``name``, ``description``, ``max_results`` (its ceiling),
    ``extra_options`` (its own schema entries) and ``declines`` — standard
    options it cannot honour, each with the reason the caller is told — and
    implements ``_search(query, opts, window, notices, cost) -> list[dict]``,
    returning results in the common record and, where its answer reports what
    the call cost, filling ``cost`` with ``usd``, ``basis`` and ``usage`` (left
    empty, the call is priced at ``_LIST_PRICE``). ``window`` is the resolved
    ``(start, end)`` dates (each ``""`` when open) and ``opts["recency"]`` the
    window word when no explicit date narrowed it, for providers that take the
    word natively.
    """

    name        = "search"
    description = "Search the web and return a plain-text numbered list of results."
    # It reads the web and changes nothing anywhere.
    risk        = "read"
    #: The provider's own ceiling on results per call.
    max_results: int = 20
    #: Provider-only option schemas, added beside the standard ones.
    extra_options: dict = {}
    #: Standard options this provider cannot honour → why.
    declines: dict = {}
    #: The provider's key variable; a tuple here so the base names none.
    _ENV_KEY: "str | tuple" = ()
    _PROVIDER: str = ""
    #: USD a call at the published rate, for a provider whose answer reports no cost.
    _LIST_PRICE: float = 0.0
    #: Where that rate was read, and when.
    _PRICE_SOURCE: str = ""

    def __init__(self, api_key: "str | None" = None, *, name: "str | None" = None,
                 description: "str | None" = None) -> None:
        key = api_key or (os.environ.get(self._ENV_KEY) if isinstance(self._ENV_KEY, str) else None)
        if not key:
            raise ValueError(f"No {self._PROVIDER} API key found. Pass api_key= or set the "
                             f"{self._ENV_KEY} environment variable.")
        self._api_key = key
        if name:
            self.name = name
        if description:
            self.description = description
        self._http = make_http(timeout=SEARCH_TIMEOUT, retries=SEARCH_RETRIES)

    parameters = {
        "type": "object",
        "properties": {
            "input":   {"type": "string", "description": "The search query, in natural language."},
            "options": {"type": "object", "description": "Optional search filters.",
                        "properties": dict(STANDARD_OPTIONS)},
        },
        "required": ["input"],
    }

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        # Every standard option is declared on every provider — a declined one
        # says so in its description, and asking for it is answered with a notice.
        props = {k: (v if k not in cls.declines else
                     {**v, "description": f"{v['description']} Not applied here: {cls.declines[k]}."})
                 for k, v in STANDARD_OPTIONS.items()}
        props.update(cls.extra_options)
        cls.parameters = {
            "type": "object",
            "properties": {
                "input":   {"type": "string", "description": "The search query, in natural language."},
                "options": {"type": "object", "description": "Optional search filters.",
                            "properties": props},
            },
            "required": ["input"],
        }

    # ------------------------------------------------------------------

    def run(self, input: str, options: "dict | None" = None) -> Record:  # type: ignore[override]
        """
        Search for *input*. The result is a :class:`Record`: the fields below
        for code, and the numbered list of :meth:`render` for a model or a
        prompt.
        """
        if not isinstance(input, str) or not input.strip():
            raise ValueError(f"{type(self).__name__} needs a query — it was empty")
        opts = dict(options or {})
        notices: list = []
        for key in [k for k in opts if k in self.declines and opts[k] not in (None, "", [])]:
            self._notice(notices, f"{key} is not supported by {self.name} and was not applied: "
                                  f"{self.declines[key]}")
            opts.pop(key)
        count = int(opts.get("max_results") or 10)
        if count > self.max_results:
            self._notice(notices, f"max_results {count} is above {self.name}'s ceiling; "
                                  f"{self.max_results} were asked for")
        opts["max_results"] = max(1, min(count, self.max_results))
        window = self._window(opts)
        cost: dict = {}
        results = self._search(input.strip(), opts, window, notices, cost)
        if "usd" not in cost:
            cost.update(usd=self._LIST_PRICE, basis=f"published rate ({self._PRICE_SOURCE})")
        return Record({"query": input.strip(), "provider": self.name, "results": results, "notices": notices,
                       "cost": round(cost["usd"], 6), "cost_basis": cost["basis"], "usage": cost.get("usage", {})},
                      self.render)

    def search(self, input: str, options: "dict | None" = None) -> Record:
        """The same as :meth:`run` — kept from 3.2, when ``run()`` returned the text."""
        return self.run(input, options)

    def render(self, found: dict) -> str:
        """What a model reads: the results as a numbered list, then any notes."""
        return self.format(found)

    # ------------------------------------------------------------------

    @staticmethod
    def format(found: dict) -> str:
        results = found["results"]
        lines = [f'Search results for "{found["query"]}" ({len(results)} results):\n']
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r['title'] or '(untitled)'}")
            lines.append(f"    URL: {r['url']}")
            meta = ([f"Published: {r['date']}"] if r["date"] else []) + \
                   ([f"Author: {r['author']}"] if r["author"] else [])
            if meta:
                lines.append("    " + " · ".join(meta))
            # one line each: the page's own line breaks would break the list
            lines += ["    · " + " ".join(s.split()) for s in r["snippets"] if s and s.strip()]
            if r["text"]:
                lines.append("    Text: " + " ".join(r["text"].split()))
            lines.append("")
        for n in found["notices"]:
            lines.append(f"Note: {n}")
        return "\n".join(lines).rstrip()

    @staticmethod
    def record(title="", url="", date_="", author="", snippets=(), text="") -> dict:
        """One result in the common shape — every key present."""
        return {"title": (title or "").strip(), "url": url or "", "date": day(date_),
                "author": (author or "").strip() if isinstance(author, str) else "",
                "snippets": [s for s in snippets if isinstance(s, str) and s.strip()],
                "text": text or ""}

    def _window(self, opts: dict, now: "datetime | None" = None) -> tuple:
        """The (start, end) days the options ask for; the later start wins."""
        start = _check_day("start_date", opts["start_date"]) if opts.get("start_date") else ""
        end = _check_day("end_date", opts["end_date"]) if opts.get("end_date") else ""
        rec = opts.get("recency")
        if rec:
            if rec not in RECENCY:
                raise ValueError(f"recency must be one of {', '.join(RECENCY)}; got {rec!r}")
            since = ((now or datetime.now(timezone.utc)) - RECENCY[rec]).date().isoformat()
            if start and start >= since:
                opts.pop("recency")          # the explicit date is the tighter bound
            else:
                start = since
        if start and end and start > end:
            raise ValueError(f"start_date {start} is after end_date {end}")
        return start, end

    def _notice(self, notices: list, text: str) -> None:
        notices.append(text)
        warnings.warn(f"{self.name}: {text}", stacklevel=4)

    def _search(self, query: str, opts: dict, window: tuple, notices: list, cost: dict) -> list:  # pragma: no cover
        raise NotImplementedError(f"{type(self).__name__} must implement _search()")
