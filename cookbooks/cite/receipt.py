"""What a run cost, one row per stage. A call with no price is counted, never priced at $0."""

from contextlib import contextmanager


class Receipt:
    def __init__(self):
        self.rows, self._stage = {}, "run"

    @contextmanager
    def stage(self, name):
        previous, self._stage = self._stage, name
        self.rows.setdefault(name, {"calls": 0, "in": 0, "out": 0, "cost": 0.0, "unpriced": 0})
        try:
            yield
        finally:
            self._stage = previous

    def add(self, usage, calls=1):
        """A Skill's last_usage, or a tool's {"input_tokens", "output_tokens", "cost"}."""
        get = usage.get if isinstance(usage, dict) else lambda k: getattr(usage, k, None)
        row = self.rows.setdefault(self._stage, {"calls": 0, "in": 0, "out": 0, "cost": 0.0, "unpriced": 0})
        row["calls"] += calls
        row["in"] += get("input_tokens") or 0
        row["out"] += get("output_tokens") or 0
        if get("cost") is None:
            row["unpriced"] += calls
        else:
            row["cost"] += get("cost")

    def watch(self, embedder):
        """Count every call an embedder makes, inside a store's upsert and query too."""
        inner = embedder.embed

        def embed(texts, **kwargs):
            result = inner(texts, **kwargs)
            self.add({"input_tokens": result.total_tokens, "cost": getattr(result, "cost", None)})
            return result
        embedder.embed = embed
        return embedder

    def total(self):
        return sum(r["cost"] for r in self.rows.values())

    def print(self):
        print(f"\n{'stage':<10}{'calls':>7}{'in tok':>10}{'out tok':>10}{'cost':>10}  unpriced")
        for name, r in [*self.rows.items(), ("total", {k: sum(r[k] for r in self.rows.values())
                                                       for k in ("calls", "in", "out", "cost", "unpriced")})]:
            cost = f"${r['cost']:.4f}" if r["calls"] > r["unpriced"] else "—"
            print(f"{name:<10}{r['calls']:>7}{r['in']:>10,}{r['out']:>10,}{cost:>10}  {r['unpriced'] or ''}")
