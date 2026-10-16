"""The shop: a refund policy and fifty orders, the same on every run."""

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

DATA = Path(__file__).parent / "data"

POLICY = """# Refund policy

We refund an order when the item arrived damaged or defective, or never
arrived, and the customer asks within 30 days of delivery.

- A refund is for the amount paid, or less if the customer asks for less.
- An order is refunded once. A second request for the same order is refused,
  politely, with the date of the first refund if we have it.
- Refunds up to $50 are issued at once by support.
- Refunds over $50 need a manager's approval. Tell the customer it has been
  sent for approval; the manager may answer later.
- When a manager refuses, tell the customer the manager's reason as given.
- An order that is still on its way is not refunded; give its status.
- Never promise a refund before it has been issued.
"""

#: The orders the scenarios use: (customer, item, amount, status, delivered_on).
SCENARIO_ORDERS = {
    "A-1003": ("Ana Ruiz", "desk lamp", 42.00, "shipped", None),
    "A-1007": ("Ben Okafor", "ceramic mug set", 30.00, "delivered", "2026-10-09"),
    "A-1031": ("Chloe Martin", "office chair", 300.00, "delivered", "2026-10-06"),
    "A-1042": ("Dev Patel", "standing desk", 120.00, "delivered", "2026-10-02"),
}
ITEMS = ["notebook", "pen set", "monitor arm", "keyboard", "headset", "backpack", "webcam"]
NAMES = ["Eva", "Femi", "Gus", "Hana", "Ivo", "Jun", "Kai", "Lea", "Mo", "Noor"]


def build() -> Path:
    """Write data/policy.md and a fresh data/shop.db; return the database's path."""
    DATA.mkdir(exist_ok=True)
    (DATA / "policy.md").write_text(POLICY)
    path = DATA / "shop.db"
    path.unlink(missing_ok=True)
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE orders (id TEXT PRIMARY KEY, customer TEXT, item TEXT, amount REAL, "
               "status TEXT, delivered_on TEXT, refunded INTEGER DEFAULT 0)")
    rng = random.Random(7)
    for n in range(1000, 1050):
        order = f"A-{n}"
        delivered = date(2026, 10, 1) + timedelta(days=rng.randrange(14))
        row = SCENARIO_ORDERS.get(order) or (
            f"{rng.choice(NAMES)} {rng.choice('ABCDEFG')}.", rng.choice(ITEMS),
            round(rng.uniform(8, 400), 2), "delivered", delivered.isoformat())
        db.execute("INSERT INTO orders (id, customer, item, amount, status, delivered_on) "
                   "VALUES (?, ?, ?, ?, ?, ?)", (order, *row))
    db.commit()
    for order, row in SCENARIO_ORDERS.items():
        assert db.execute("SELECT amount FROM orders WHERE id = ?", (order,)).fetchone()[0] == row[2]
    db.close()
    return path


if __name__ == "__main__":
    print(build())
