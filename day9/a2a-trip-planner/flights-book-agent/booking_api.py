# ---------------------------------------------------------------------------
# booking_api.py  ·  the Booking Agent's TOOL (issue a confirmation)
# ---------------------------------------------------------------------------
# Booking a *real* seat needs a live offer id + a payment (e.g. Duffel orders),
# which no free/keyless path gives — so, same honesty as the flights search
# tool, this issues a deterministic SAMPLE confirmation (a PNR) and says so.
# It only runs once all required fields are present; the agent handles the
# multi-turn "still missing something" logic (that's the input-required lesson).
# ---------------------------------------------------------------------------

import zlib
import random

_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"   # PNR-style, no ambiguous chars


def _pnr(seed_str: str) -> str:
    rnd = random.Random(zlib.crc32(seed_str.encode()))
    return "".join(rnd.choice(_ALPHABET) for _ in range(6))


async def book_flight(fields: dict) -> dict:
    """Issue a (sample) booking confirmation for a fully-specified request."""
    passenger = (fields.get("passenger") or "").strip()
    flight = (fields.get("flight") or "").strip().upper()
    seat_pref = (fields.get("seat") or "").strip().lower()

    rnd = random.Random(zlib.crc32((passenger + flight).encode()))
    row = rnd.randint(8, 42)
    col = "A" if seat_pref.startswith("w") else "C" if seat_pref.startswith("a") else rnd.choice("ABCDEF")

    return {
        "ok": True, "source": "sample",
        "pnr": _pnr(passenger + flight),
        "passenger": passenger,
        "flight": flight,
        "seat": f"{row}{col}",
        "seat_pref": fields.get("seat", ""),
        "dateOfBirth": fields.get("dateOfBirth", ""),
        "boarding_group": rnd.choice(["A", "B", "C"]),
        "note": "sample confirmation — no real seat was reserved",
    }
