"""
Reconciliation cases with planted ground truth — the oracle for the benchmark.

A case is a contract, a handful of invoices, and the delivery notes behind
them. Some invoices are clean. Others carry defects that *we put there*, so
the answer key is exact and free: no judge model, no hand labelling.

The defects span three depths, and that ladder is the point — it is what
separates one agent strategy from another:

  local    visible inside a single document (bad arithmetic, wrong total)
  cross    needs a second document (contract rate, delivery note quantity)
  derived  needs a conditional clause plus date arithmetic (missing penalty)

Run it on its own to see a case and check the generator against itself:

    python generate.py            # print one case
    python generate.py --check    # 200 cases, assert the oracle is sound
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta

# ── The world ────────────────────────────────────────────────────────────────

CATALOG = [
    ("PAL-STD",  "Standard pallet handling",     42.50),
    ("PAL-COLD", "Refrigerated pallet handling", 68.00),
    ("CRT-SML",  "Small crate handling",         11.25),
    ("CRT-LRG",  "Large crate handling",         19.80),
    ("STG-DAY",  "Warehouse storage, per day",    3.40),
    ("DLV-URB",  "Urban delivery run",           95.00),
    ("DLV-REG",  "Regional delivery run",       180.00),
    ("HND-OVR",  "Oversize handling surcharge",  55.00),
]
RATES = {code: rate for code, _, rate in CATALOG}

TERM_START = date(2026, 1, 1)
TERM_END   = date(2026, 6, 30)
CURRENCY   = "EUR"
LATE_DAYS  = 3      # clause 7.2 tolerance
PENALTY    = 0.05   # clause 7.2 credit

#: kind -> (depth, what it means). The list is closed and is given to the
#: agent: knowing *which faults exist* is not knowing *which invoice has one*,
#: and every strategy gets the same head start.
KINDS = {
    "rate_mismatch":     ("cross",   "unit price differs from the contracted rate"),
    "quantity_mismatch": ("cross",   "invoiced quantity differs from the delivery note"),
    "date_outside_term": ("cross",   "service period falls outside the contract term"),
    "penalty_missing":   ("derived", "delivery was late under clause 7.2 but no penalty credit was applied"),
    "duplicate_line":    ("local",   "the same charge appears twice"),
    "arithmetic_error":  ("local",   "line total does not equal quantity x unit price"),
    "total_mismatch":    ("local",   "invoice total does not equal the sum of its lines"),
    "currency_mismatch": ("local",   "invoice is issued in a currency other than the contract currency"),
}
LINE_KINDS = ("rate_mismatch", "quantity_mismatch", "arithmetic_error",
              "duplicate_line", "penalty_missing")
DOC_KINDS  = ("total_mismatch", "date_outside_term", "currency_mismatch")


#: Documents are rendered to stay under this. The Agent renders memory values
#: into its prompt truncated at 500 characters, so a longer invoice reaches the
#: reasoning step with its last lines and its TOTAL cut off — and no strategy
#: can reconcile a table it cannot see. Measured first at 658-904 characters an
#: invoice, which handed the win to the one approach that skips Agent memory.
#: The ceiling is asserted in ``_check``; raise it only with that in mind.
DOC_LIMIT = 500


@dataclass
class Case:
    """One reconciliation job plus its answer key."""
    seed:      int
    documents: dict            # name -> rendered text, contract included
    truth:     list            # [{kind, document, lines, detail}]

    @property
    def contract(self) -> str:
        """Both contract parts, for a consumer that takes everything at once."""
        return "\n".join(self.documents[n] for n in
                         ("CONTRACT-RATES", "CONTRACT-TERMS"))

    def depth_counts(self) -> dict:
        out: dict = {}
        for t in self.truth:
            out[KINDS[t["kind"]][0]] = out.get(KINDS[t["kind"]][0], 0) + 1
        return out


# ── Building one case ────────────────────────────────────────────────────────

def _invoice(rng, idx: int) -> dict:
    """A clean invoice and its delivery note — every number consistent."""
    codes = rng.sample([c for c, _, _ in CATALOG], rng.randint(3, 6))
    start = TERM_START + timedelta(days=rng.randint(0, 120))
    lines = []
    for code in codes:
        agreed = start + timedelta(days=rng.randint(0, 20))
        lines.append({
            "code": code,
            "desc": next(d for c, d, _ in CATALOG if c == code),
            "qty": rng.randint(1, 25),
            "price": RATES[code],
            "note_qty": None,                                  # filled below
            "agreed": agreed,
            # inside tolerance, so a late delivery only exists where planted
            "delivered": agreed + timedelta(days=rng.randint(0, LATE_DAYS)),
        })
    for ln in lines:
        ln["note_qty"] = ln["qty"]
    return {
        "id":       f"INV-2026-{1000 + idx * 7 + rng.randint(0, 6)}",
        "start":    start,
        "end":      start + timedelta(days=29),
        "currency": CURRENCY,
        "lines":    lines,
        "total":    None,                                      # settled later
    }


def _settle(inv: dict) -> None:
    """Recompute every line total from qty x price, and the invoice total."""
    for ln in inv["lines"]:
        ln["total"] = round(ln["qty"] * ln["price"], 2)
    inv["total"] = round(sum(ln["total"] for ln in inv["lines"]), 2)


def make_case(seed: int, invoices: int = 3, defects: int = 4) -> Case:
    """
    Build a case with exactly *defects* planted faults across *invoices*.

    Defects are applied in an order that keeps every other number consistent,
    so each fault has exactly one symptom. Without that discipline a wrong unit
    price would also read as bad arithmetic and the answer key would be a lie.
    """
    rng  = random.Random(seed)
    invs = [_invoice(rng, i) for i in range(invoices)]
    for inv in invs:
        _settle(inv)

    # Choose where the faults go. A line carries at most one fault, and a
    # document-level fault appears at most once per document.
    slots = [(i, ln) for i, inv in enumerate(invs) for ln in range(len(inv["lines"]))]
    rng.shuffle(slots)
    # Draw kinds without replacement, reshuffling only once the deck runs out.
    # Picking each independently clusters badly — four defects came back as two
    # currency errors and two quantity errors, and the depth ladder that makes
    # this benchmark discriminate went untested.
    deck: list = []
    while len(deck) < defects:
        batch = list(KINDS)
        rng.shuffle(batch)
        deck += batch
    plan: list = []
    used_docs: set = set()
    for kind in deck[:defects]:
        if kind in DOC_KINDS:
            free = [i for i in range(invoices) if (i, kind) not in used_docs]
            if not free:
                continue
            i = rng.choice(free)
            used_docs.add((i, kind))
            plan.append((kind, i, None))
        else:
            if not slots:
                continue
            i, ln = slots.pop()
            plan.append((kind, i, ln))

    truth: list = []

    def _plant(kind, i, lines, detail):
        truth.append({"kind": kind, "document": invs[i]["id"],
                      "lines": lines, "detail": detail})

    # ── 1. Value changes. Totals are re-settled afterwards, so the only
    #       symptom is the mismatch against the *other* document.
    for kind, i, ln in plan:
        inv = invs[i]
        if kind == "rate_mismatch":
            line = inv["lines"][ln]
            line["price"] = round(line["price"] * rng.choice([1.15, 1.25, 0.85]), 2)
            _plant(kind, i, [ln + 1], f"{line['code']} billed at {line['price']}, "
                                      f"contract says {RATES[line['code']]}")
        elif kind == "quantity_mismatch":
            line = inv["lines"][ln]
            line["qty"] = line["note_qty"] + rng.randint(1, 5)
            _plant(kind, i, [ln + 1], f"{line['code']} billed {line['qty']}, "
                                      f"delivery note says {line['note_qty']}")
        elif kind == "duplicate_line":
            copy = dict(inv["lines"][ln])
            copy["note_qty"] = 0          # nothing extra was ever delivered
            inv["lines"].insert(ln + 1, copy)
            # either occurrence is a fair answer; the fault is the pair
            _plant(kind, i, [ln + 1, ln + 2], f"{copy['code']} appears twice")
            # every later slot in this invoice shifted down by one
            slots = [(a, b + 1) if a == i and b > ln else (a, b) for a, b in slots]
        elif kind == "penalty_missing":
            line = inv["lines"][ln]
            late = rng.randint(LATE_DAYS + 1, 9)
            line["delivered"] = line["agreed"] + timedelta(days=late)
            _plant(kind, i, [ln + 1, None],
                   f"{line['code']} delivered {late} days late, no 5% credit line")

    for inv in invs:
        _settle(inv)

    # ── 2. Faults in the stated numbers themselves. Applied after settling so
    #       they are not wiped, and the invoice total follows the *stated*
    #       line totals so a bad line does not also break the sum.
    for kind, i, ln in plan:
        inv = invs[i]
        if kind == "arithmetic_error":
            line = inv["lines"][ln]
            line["total"] = round(line["total"] + rng.choice([-1, 1]) * rng.randint(5, 90), 2)
            inv["total"] = round(sum(x["total"] for x in inv["lines"]), 2)
            _plant(kind, i, [ln + 1], f"line total {line['total']} != "
                                      f"{line['qty']} x {line['price']}")

    # ── 3. Header-level faults. They touch nothing else.
    for kind, i, ln in plan:
        inv = invs[i]
        if kind == "total_mismatch":
            inv["total"] = round(inv["total"] + rng.choice([-1, 1]) * rng.randint(20, 200), 2)
            _plant(kind, i, [], f"stated total {inv['total']} != sum of lines")
        elif kind == "date_outside_term":
            inv["start"] = TERM_END - timedelta(days=rng.randint(0, 5))
            inv["end"]   = inv["start"] + timedelta(days=29)
            _plant(kind, i, [], f"period ends {inv['end']}, contract ends {TERM_END}")
        elif kind == "currency_mismatch":
            inv["currency"] = rng.choice(["USD", "GBP", "CHF"])
            _plant(kind, i, [], f"issued in {inv['currency']}, contract is {CURRENCY}")

    docs = {"CONTRACT-RATES": _render_rates(), "CONTRACT-TERMS": _render_terms()}
    for inv in invs:
        docs[inv["id"]]                     = _render_invoice(inv)
        docs[inv["id"].replace("INV", "DN")] = _render_note(inv)
    return Case(seed=seed, documents=docs, truth=truth)


# ── Rendering ────────────────────────────────────────────────────────────────
#
# Terse on purpose — see DOC_LIMIT. Boilerplate that repeats on every document
# (supplier, client, contract id) is dropped: it costs characters and carries
# no signal, since every document in a case belongs to the same agreement.

def _render_rates() -> str:
    rows = "\n".join(f"  {c:<9} {r:>7,.2f}  {d}" for c, d, r in CATALOG)
    return f"CONTRACT SVC-2026-0417 - clause 4.1, agreed unit rates in {CURRENCY}\n{rows}\n"


def _render_terms() -> str:
    return f"""\
CONTRACT SVC-2026-0417 - terms
Northwind Logistics BV (supplier) to Acme Manufacturing GmbH (client).
4.1  Unit rates are listed in CONTRACT-RATES.
7.2  Where goods arrive more than {LATE_DAYS} days after the agreed date, a
     penalty of {PENALTY:.0%} of that line's value is due to the client and must
     appear on the invoice as a separate credit line.
9.1  All amounts are invoiced in {CURRENCY}.
9.4  Term runs {TERM_START} to {TERM_END}. A service period may not fall
     outside it.
"""


def _render_invoice(inv: dict) -> str:
    head = (f"INVOICE {inv['id']}  period {inv['start']}..{inv['end']}  "
            f"{inv['currency']}\n"
            f"    #  code       qty     price     total\n")
    body = "\n".join(
        f"  {n:>3}  {l['code']:<9} {l['qty']:>4} {l['price']:>9,.2f} {l['total']:>9,.2f}"
        for n, l in enumerate(inv["lines"], 1))
    return f"{head}{body}\n       TOTAL {inv['total']:>26,.2f}\n"


def _render_note(inv: dict) -> str:
    head = (f"DELIVERY NOTE {inv['id'].replace('INV', 'DN')} for {inv['id']}\n"
            f"  code       delivered  agreed date  arrived\n")
    seen: set = set()
    rows = []
    for l in inv["lines"]:
        if l["code"] in seen:      # a duplicated charge was delivered once
            continue
        seen.add(l["code"])
        rows.append(f"  {l['code']:<9} {l['note_qty']:>10}  {l['agreed']}   {l['delivered']}")
    return head + "\n".join(rows) + "\n"


# ── Scoring ──────────────────────────────────────────────────────────────────

def score(case: Case, reported: list) -> dict:
    """
    Match reported findings against the answer key.

    A **line-level** finding matches on location — document plus line — not on
    the label. That is deliberate: the actionable output of a reconciliation is
    "line 3 of this invoice is wrong", and arguing over whether an over-billed
    duplicate is a duplicate or a quantity error measures taxonomy, not
    competence. Whether the label was also right is reported separately as
    ``kind_agreement``.

    A **document-level** finding has no line to point at, so there the label is
    the location and must match. Two header faults on one invoice are otherwise
    indistinguishable, and the answer key would silently lose one.
    """
    matched: dict = {}
    used:    set  = set()
    for f in reported:
        doc  = str(f.get("document") or "").strip()
        kind = str(f.get("kind") or "").strip()
        line = f.get("line")
        line = int(line) if isinstance(line, (int, float, str)) and str(line).strip().isdigit() else None
        # Invoice lines are 1-based, so 0 can only mean "no line" — the header.
        # Native tool schemas invite this encoding: with `line` declared as an
        # integer parameter, models fill "not applicable" as 0 rather than
        # omitting the field. Scoring 0 as a line number turned one correct
        # header finding into a miss AND a false positive at once.
        if line == 0:
            line = None
        key  = (doc, line) if line is not None else (doc, kind)
        if key in used:                        # same spot reported twice
            continue
        used.add(key)
        # Prefer a truth entry whose label also agrees. Without that, a
        # header-level report could be absorbed by whichever header fault the
        # answer key happens to list first, and the label would stop mattering.
        for want_kind in (True, False):
            hit = None
            for i, t in enumerate(case.truth):
                if i in matched or t["document"] != doc:
                    continue
                if want_kind and kind != t["kind"]:
                    continue
                ok = (line in t["lines"]) if t["lines"] else (line is None and kind == t["kind"])
                if ok:
                    hit = i
                    break
            if hit is not None:
                matched[hit] = f
                break

    planted = len(case.truth)
    found   = len(matched)
    agreed  = sum(1 for i, f in matched.items()
                  if str(f.get("kind", "")).strip() == case.truth[i]["kind"])
    fp      = len(used) - found
    by_depth: dict = {}
    for i, t in enumerate(case.truth):
        d = by_depth.setdefault(KINDS[t["kind"]][0], [0, 0])
        d[1] += 1
        if i in matched:
            d[0] += 1
    return {
        "planted": planted, "found": found, "false_positives": fp,
        "reported": len(used),
        # The headline. Recall alone is gameable — flag every line of every
        # invoice and it reaches 1.0 — so "did the job" means everything
        # planted was found *and* nothing was invented. One bit, per case.
        "solved":  found == planted and fp == 0,
        "recall":         found / planted if planted else 1.0,
        "precision":      found / len(used) if used else 1.0,
        "kind_agreement": agreed / found if found else 0.0,
        # counts, not ratios: a caller summing several cases must be able to
        # pool them. Handing back per-case ratios forces an average of averages,
        # which weights a case with one planted fault the same as a case with six.
        "by_depth":       {k: tuple(v) for k, v in by_depth.items()},
        "missed":         [case.truth[i] for i in range(planted) if i not in matched],
    }


# ── Self-check ───────────────────────────────────────────────────────────────

def _check(n: int = 200) -> None:
    """
    Verify the oracle. The failure this guards against is a generator that
    plants one fault and accidentally creates a second: then the answer key is
    incomplete and every agent is scored against a wrong number.
    """
    for seed in range(n):
        case = make_case(seed, invoices=3, defects=4)
        # every planted fault points at a document that exists
        for t in case.truth:
            assert t["document"] in case.documents, (seed, t)
        # no document may outgrow what the Agent can see (see DOC_LIMIT)
        for name, text in case.documents.items():
            assert len(text) <= DOC_LIMIT, (seed, name, len(text))
        # a clean case must be perfectly consistent
        clean = make_case(seed, invoices=3, defects=0)
        assert not clean.truth
        for name, text in clean.documents.items():
            if not name.startswith("INV"):
                continue
            total = 0.0
            for line in text.splitlines():
                parts = line.split()
                if len(parts) == 5 and parts[0].isdigit():
                    qty, price, ltot = (float(p.replace(",", "")) for p in parts[-3:])
                    assert abs(qty * price - ltot) < 0.01, (seed, name, line)
                    total += ltot
                if parts[:1] == ["TOTAL"]:
                    assert abs(total - float(parts[1].replace(",", ""))) < 0.01, (seed, name)
        # a perfect answer scores 1.0; an empty one scores 0.0
        perfect = [{"kind": t["kind"], "document": t["document"],
                    "line": (t["lines"] or [None])[0]} for t in case.truth]
        s = score(case, perfect)
        assert s["recall"] == 1.0 and s["false_positives"] == 0, (seed, s)
        assert score(case, [])["recall"] == 0.0 or not case.truth
    print(f"ok - {n} cases: answer key sound, clean invoices internally consistent")


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        _check()
    else:
        case = make_case(7, invoices=3, defects=4)
        for name, text in case.documents.items():
            print(f"{text}[{len(text)} chars]\n")
        print("ANSWER KEY")
        for t in case.truth:
            print(f"  {t['kind']:<18} {t['document']} lines={t['lines']}  {t['detail']}")
