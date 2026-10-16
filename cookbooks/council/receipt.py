"""What a run cost, one row per stage. A call with no price is counted, never priced at $0."""

from contextlib import contextmanager


class Receipt:
    def __init__(self):
        self.rows, self._stage = {}, "run"

    @contextmanager
    def stage(self, name):
        previous, self._stage = self._stage, name
        self.rows.setdefault(name, {"calls": 0, "tokens": 0, "cost": 0.0, "unpriced": 0})
        try:
            yield
        finally:
            self._stage = previous

    def add(self, usage, calls=1):
        """A Skill's last_usage, or {"tokens" or "input_tokens"/"output_tokens", "cost"}."""
        get = usage.get if isinstance(usage, dict) else lambda k: getattr(usage, k, None)
        row = self.rows.setdefault(self._stage, {"calls": 0, "tokens": 0, "cost": 0.0, "unpriced": 0})
        row["calls"] += calls
        row["tokens"] += (get("tokens") or get("total_tokens")
                          or (get("input_tokens") or 0) + (get("output_tokens") or 0))
        if get("cost") is None:
            row["unpriced"] += calls
        else:
            row["cost"] += get("cost")

    def hook(self):
        """For hooks=[...]: every model call an agent, a chain or a skill makes."""
        def on(event):
            if event.type == "llm_call.ended":
                self.add({"tokens": event.usage, "cost": event.cost})
        return on

    def total(self):
        return sum(r["cost"] for r in self.rows.values())

    def print(self):
        print(f"\n{'stage':<22}{'calls':>6}{'tokens':>10}{'cost':>10}  unpriced")
        total = {k: sum(r[k] for r in self.rows.values()) for k in ("calls", "tokens", "cost", "unpriced")}
        for name, r in [*self.rows.items(), ("total", total)]:
            cost = f"${r['cost']:.4f}" if r["calls"] > r["unpriced"] else "—"
            print(f"{name:<22}{r['calls']:>6}{r['tokens']:>10,}{cost:>10}  {r['unpriced'] or ''}")
