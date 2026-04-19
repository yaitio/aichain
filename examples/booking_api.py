"""
examples/booking_api.py
========================

RestApiTool — Restful Booker API demo
======================================

Demonstrates how ``RestApiTool`` turns any REST API into pipeline-ready tools
with no boilerplate.  Uses the public `Restful Booker <https://restful-booker.herokuapp.com/>`_
test API to run a complete booking lifecycle through a single Chain:

    POST /auth          → token
    POST /booking       → bookingid
    GET  /booking/{id}  → booking details
    PATCH /booking/{id} → partial update (auth required)
    DELETE /booking/{id}→ cleanup    (auth required)

Key patterns this example shows
---------------------------------

1. **Declarative endpoint config** — each endpoint is defined once in 4–8 lines.
   Method, URL, which fields go into the path / query / body, and the auth
   strategy are all constructor arguments.  No HTTP boilerplate, no manual
   header assembly.

2. **Automatic variable flow** — a Chain passes every accumulated variable
   whose name matches a declared parameter.  ``token`` from Step 1 and
   ``bookingid`` from Step 2 arrive automatically at Steps 4–5 without any
   input_map wiring.

3. **Dict responses are auto-merged** — ``POST /auth`` returns ``{"token": "…"}``
   and ``POST /booking`` returns ``{"bookingid": N, "booking": {…}}``.  Both
   are dict responses, so all their keys are merged into the Chain's accumulated
   dict and become variables for downstream steps.

4. **input_map for renaming** — Step 4 (PATCH) uses ``input_map`` to pass
   ``accumulated["updated_needs"]`` as the ``additionalneeds`` body field,
   demonstrating how to pass a differently-named variable.

5. **No API keys required** — the Restful Booker API is a public test API.
   Run with: ``python examples/booking_api.py``
"""

from __future__ import annotations

import os
import sys

# ── Resolve library root ──────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.path.normpath(os.path.join(_HERE, ".."))
sys.path.insert(0, _LIB_DIR)

try:
    import dotenv
    dotenv.load_dotenv(os.path.join(_LIB_DIR, ".env"))
except ImportError:
    pass

from chain import Chain
from tools import RestApiTool


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint definitions
# ─────────────────────────────────────────────────────────────────────────────
# Each RestApiTool instance is one configured endpoint.
# Declare them once at module level — reuse anywhere.

BASE = "https://restful-booker.herokuapp.com"

# POST /auth — get a session token
authenticate = RestApiTool(
    name        = "authenticate",
    description = "Authenticate with username/password and receive a session token.",
    method      = "POST",
    url         = f"{BASE}/auth",
    body_params = ["username", "password"],
    required_params = ["username", "password"],
)

# GET /booking — list all bookings (with optional filters)
list_bookings = RestApiTool(
    name        = "list_bookings",
    description = "List all booking IDs, optionally filtered by guest name or dates.",
    method      = "GET",
    url         = f"{BASE}/booking",
    query_params = ["firstname", "lastname", "checkin", "checkout"],
)

# POST /booking — create a new booking
create_booking = RestApiTool(
    name        = "create_booking",
    description = "Create a new hotel booking. Returns bookingid and full booking object.",
    method      = "POST",
    url         = f"{BASE}/booking",
    body_params = [
        "firstname", "lastname", "totalprice",
        "depositpaid", "bookingdates", "additionalneeds",
    ],
    required_params = ["firstname", "lastname", "totalprice", "depositpaid", "bookingdates"],
    headers     = {"Accept": "application/json"},
)

# GET /booking/{bookingid} — retrieve booking details
get_booking = RestApiTool(
    name        = "get_booking",
    description = "Retrieve full details for a booking by its ID.",
    method      = "GET",
    url         = f"{BASE}/booking/{{bookingid}}",
    path_params = ["bookingid"],
    required_params = ["bookingid"],
)

# PATCH /booking/{bookingid} — partial update (auth required)
# token_field="token" means: read the token from data["token"]
# (which arrives automatically from the accumulated "token" variable set by Step 1)
patch_booking = RestApiTool(
    name        = "patch_booking",
    description = "Partially update a booking (auth required). Only supplied fields are changed.",
    method      = "PATCH",
    url         = f"{BASE}/booking/{{bookingid}}",
    path_params = ["bookingid"],
    body_params = ["firstname", "lastname", "totalprice",
                   "depositpaid", "additionalneeds"],
    required_params = ["bookingid", "token"],
    auth        = {"type": "cookie", "cookie_name": "token", "token_field": "token"},
)

# DELETE /booking/{bookingid} — delete (auth required)
delete_booking = RestApiTool(
    name        = "delete_booking",
    description = "Delete a booking by ID (auth required). Returns HTTP 201 on success.",
    method      = "DELETE",
    url         = f"{BASE}/booking/{{bookingid}}",
    path_params = ["bookingid"],
    required_params = ["bookingid", "token"],
    auth        = {"type": "cookie", "cookie_name": "token", "token_field": "token"},
)


# ─────────────────────────────────────────────────────────────────────────────
# Chain — full booking lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def build_lifecycle_chain() -> Chain:
    """
    Build a 5-step Chain that runs the full booking lifecycle.

    Variable flow (automatically via dict-merge and name matching):

        Step 1  POST /auth
                body:     username, password  (from initial vars)
                response: {"token": "abc"}
                          ↳ token → accumulated

        Step 2  POST /booking
                body:     firstname, lastname, totalprice, depositpaid,
                          bookingdates, additionalneeds  (from initial vars)
                response: {"bookingid": 42, "booking": {...}}
                          ↳ bookingid → accumulated
                          ↳ booking   → accumulated

        Step 3  GET /booking/{bookingid}
                path:     bookingid  ← accumulated (from Step 2)
                response: {firstname, lastname, totalprice, ...}
                          ↳ all fields → accumulated

        Step 4  PATCH /booking/{bookingid}
                path:     bookingid     ← accumulated (from Step 2)
                body:     additionalneeds ← input_map maps "updated_needs"
                          token (auth)  ← accumulated (from Step 1)
                response: updated booking dict → accumulated

        Step 5  DELETE /booking/{bookingid}
                path:     bookingid  ← accumulated (from Step 2)
                auth:     token      ← accumulated (from Step 1)
                response: "Created"  (201 text response)
    """
    return Chain(
        steps = [
            # Step 1 — authenticate → {token} merged into accumulated
            (authenticate, "auth_response"),

            # Step 2 — create booking → {bookingid, booking} merged
            (create_booking, "create_response"),

            # Step 3 — read it back → booking fields merged
            # bookingid comes from accumulated automatically (name match)
            (get_booking, "get_response"),

            # Step 4 — partial update
            # • bookingid and token arrive automatically (name match)
            # • input_map renames "updated_needs" → "additionalneeds"
            #   so the chain reads accumulated["updated_needs"] and passes
            #   it as the tool's "additionalneeds" body field
            (patch_booking, "patch_response",
             {"additionalneeds": "updated_needs"}),

            # Step 5 — delete
            # bookingid and token arrive automatically
            (delete_booking, "delete_response"),
        ],
        name        = "booking_lifecycle",
        description = "Authenticate → create → read → update → delete a booking.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_demo() -> None:
    print("\n── Restful Booker — booking lifecycle via RestApiTool + Chain ────────\n")

    chain = build_lifecycle_chain()

    initial_vars = {
        # Auth credentials (Restful Booker's default test account)
        "username": "admin",
        "password": "password123",

        # New booking data
        "firstname":       "James",
        "lastname":        "Brown",
        "totalprice":      250,
        "depositpaid":     True,
        "bookingdates":    {"checkin": "2025-09-01", "checkout": "2025-09-07"},
        "additionalneeds": "Breakfast",

        # Value used in the PATCH step (input_map maps this → additionalneeds)
        "updated_needs":   "Breakfast + Dinner",
    }

    print("  Initial variables:")
    for k, v in initial_vars.items():
        if k != "password":
            print(f"    {k:<20} = {v}")
    print()

    chain.run(variables=initial_vars)
    acc = chain.accumulated

    # ── Print a readable trace ─────────────────────────────────────────────
    _section("Step 1 — POST /auth")
    print(f"  token     = {acc.get('token', '(not found)')}")

    _section("Step 2 — POST /booking")
    print(f"  bookingid = {acc.get('bookingid', '(not found)')}")
    booking = acc.get("booking", {})
    if booking:
        print(f"  booking   = {booking}")

    _section("Step 3 — GET /booking/{bookingid}")
    for field in ("firstname", "lastname", "totalprice", "depositpaid",
                  "bookingdates", "additionalneeds"):
        if field in acc:
            print(f"  {field:<20} = {acc[field]}")

    _section("Step 4 — PATCH /booking/{bookingid}")
    patch = acc.get("patch_response", {})
    if patch:
        print(f"  additionalneeds = {patch.get('additionalneeds', '?')}")
        print(f"  (was: 'Breakfast', now: '{acc.get('updated_needs')}')")

    _section("Step 5 — DELETE /booking/{bookingid}")
    print(f"  result = {acc.get('delete_response', '(not found)')!r}")

    print("\n── Done ─────────────────────────────────────────────────────────────\n")


def _section(title: str) -> None:
    print(f"\n  {'─' * 60}")
    print(f"  {title}")
    print(f"  {'─' * 60}")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone usage examples (direct tool calls, no Chain)
# ─────────────────────────────────────────────────────────────────────────────

def run_standalone_examples() -> None:
    """
    Show RestApiTool used directly (outside a Chain).
    Handy for one-off calls or debugging individual endpoints.
    """
    print("\n── Standalone examples ──────────────────────────────────────────────\n")

    # 1. Health check — no parameters, no auth
    print("  GET /ping")
    result = RestApiTool(
        name="ping", description="Health check.",
        method="GET", url=f"{BASE}/ping",
    ).run()
    print(f"  → {result!r}\n")

    # 2. List bookings with a filter
    print("  GET /booking?firstname=Sally")
    result = list_bookings.run(input={"firstname": "Sally"})
    print(f"  → {result!r}\n")

    # 3. Full lifecycle in 4 direct calls
    print("  POST /auth")
    auth_resp = authenticate.run(input={"username": "admin", "password": "password123"})
    token = auth_resp.get("token") if isinstance(auth_resp, dict) else None
    print(f"  → token = {token}\n")

    print("  POST /booking")
    create_resp = create_booking.run(input={
        "firstname": "Alice", "lastname": "Smith",
        "totalprice": 175, "depositpaid": False,
        "bookingdates": {"checkin": "2025-10-01", "checkout": "2025-10-03"},
        "additionalneeds": "Late checkout",
    })
    booking_id = create_resp.get("bookingid") if isinstance(create_resp, dict) else None
    print(f"  → bookingid = {booking_id}\n")

    if booking_id:
        print(f"  DELETE /booking/{booking_id}")
        del_resp = delete_booking.run(input={"bookingid": booking_id, "token": token})
        print(f"  → {del_resp!r}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RestApiTool — Restful Booker lifecycle demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python examples/booking_api.py            # full Chain demo\n"
            "  python examples/booking_api.py --standalone  # direct tool calls\n"
            "  python examples/booking_api.py --both        # run both\n"
        ),
    )
    parser.add_argument("--standalone", action="store_true",
                        help="Run standalone (no-Chain) examples only.")
    parser.add_argument("--both", action="store_true",
                        help="Run both the Chain demo and standalone examples.")
    args = parser.parse_args()

    if args.standalone:
        run_standalone_examples()
    elif args.both:
        run_demo()
        run_standalone_examples()
    else:
        run_demo()
