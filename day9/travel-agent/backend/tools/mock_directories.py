"""
mock_directories.py — synthetic "airline directory" and "hotel directory" tools.

These stand in for real outbound APIs. They return fixed, made-up data for the
Goa scenario so the whole demo runs offline, instantly, and for free.

The important discipline: every function here has the SAME name, signature, and
return shape its live counterpart will have. Going live later means swapping the
*body* of the function — never the contract the agent depends on.
"""
from schema import ToolSpec
from tools.registry import ToolRegistry

# ── synthetic data (the Goa run) ─────────────────────────────────────────────
# Standard fares, plus cheaper fares that flexible (±3-day) dates unlock.
_FLIGHTS_STD = [
    {"airline": "Emirates", "price": 1520},
    {"airline": "Qatar",    "price": 1480},
    {"airline": "Etihad",   "price": 1610},
]
_FLIGHTS_FLEX = [
    {"airline": "Qatar",    "price": 1090},
    {"airline": "Emirates", "price": 1180},
    {"airline": "Etihad",   "price": 1320},
]
_HOTELS = [
    {"hotel": "Taj Exotica",   "price": 4100},
    {"hotel": "The Leela",     "price": 4500},
    {"hotel": "ITC Grand Goa", "price": 3650},
]

# A tiny in-memory "profile store" backing the two memory tools.
_PROFILE = {"prefers": "Emirates"}


# ── the tool functions ───────────────────────────────────────────────────────
def search_flights(from_: str = "BLR", to: str = "GOI",
                   when: str = "next month", dates: str = "") -> dict:
    """Round-trip fares. Flexible dates (pass dates='±3d') unlock cheaper fares."""
    flexible = "±3" in dates or "flex" in dates.lower()
    return {"flights": _FLIGHTS_FLEX if flexible else _FLIGHTS_STD,
            "flexible": flexible}


def find_hotels(place: str = "Goa", stars: int = 5, nights: int = 5) -> dict:
    """Hotels matching the place / star rating, for the given number of nights."""
    return {"hotels": _HOTELS, "nights": nights}


def book_flight(flight: str) -> dict:
    return {"booked": flight, "status": "confirmed"}


def book_hotel(hotel: str) -> dict:
    return {"booked": hotel, "status": "confirmed"}


def get_user_profile() -> dict:
    """SENSE: read what we remember about the user."""
    return dict(_PROFILE)


def save_user_profile(**updates) -> dict:
    """ACT: remember something for next time."""
    _PROFILE.update(updates)
    return {"saved": dict(_PROFILE)}


# ── wire every tool into a registry the loop can use ─────────────────────────
def build_registry() -> ToolRegistry:
    reg = ToolRegistry()

    reg.register(ToolSpec(
        "search_flights",
        "Search round-trip flights. Pass dates='±3d' for flexible, cheaper fares.",
        {"type": "object", "properties": {
            "from_": {"type": "string"}, "to": {"type": "string"},
            "when": {"type": "string"}, "dates": {"type": "string"}}},
    ), search_flights)

    reg.register(ToolSpec(
        "find_hotels",
        "Find hotels by place, star rating, and number of nights.",
        {"type": "object", "properties": {
            "place": {"type": "string"}, "stars": {"type": "integer"},
            "nights": {"type": "integer"}}},
    ), find_hotels)

    reg.register(ToolSpec(
        "book_flight", "Book a chosen flight.",
        {"type": "object", "properties": {"flight": {"type": "string"}},
         "required": ["flight"]},
    ), book_flight)

    reg.register(ToolSpec(
        "book_hotel", "Book a chosen hotel.",
        {"type": "object", "properties": {"hotel": {"type": "string"}},
         "required": ["hotel"]},
    ), book_hotel)

    reg.register(ToolSpec(
        "get_user_profile", "Read saved user preferences.",
        {"type": "object", "properties": {}},
    ), get_user_profile)

    reg.register(ToolSpec(
        "save_user_profile", "Save user preferences for next time.",
        {"type": "object", "properties": {}},
    ), save_user_profile)

    return reg
