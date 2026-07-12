"""
mock.py — the "No model" engine.

A real LLM reads the context and *reasons* about what to do next. The mock does
no reasoning — it follows a fixed script for the Goa scenario, keyed by how far
the conversation has already progressed. That makes runs deterministic, offline,
and free: ideal for tests and for teaching the loop itself.

The script (one decision per turn):
    1. no flights yet          -> search_flights()
    2. flights but no hotel     -> find_hotels()
    3. first combo is over cap  -> search_flights(dates='±3d')   (relax & retry)
    4. cheaper combo fits        -> book_flight + book_hotel + save_user_profile (done)

How it knows which turn it's on: it counts the tool RESULTS already in the
context. (A real model would instead read the prices and decide; the mock just
counts.)
"""
from providers.base import Provider
from schema import Message, ToolSpec, Decision, ToolCall


class MockProvider(Provider):
    name = "mock"

    def reason(self, messages: list[Message], tools: list[ToolSpec]) -> Decision:
        results = [m for m in messages if m.role == "tool"]
        flight_searches = sum(1 for m in results if m.name == "search_flights")
        hotels_found = any(m.name == "find_hotels" for m in results)

        # Turn 1 — need flights.
        if flight_searches == 0:
            return Decision(
                thought="Goal needs a round-trip flight first. Call search_flights.",
                tool_calls=[ToolCall("search_flights",
                                     {"from_": "BLR", "to": "GOI", "when": "next month"})],
            )

        # Turn 2 — have flights, still need a 5★ hotel.
        if not hotels_found:
            return Decision(
                thought="Flights are in context. The goal still needs a 5★ hotel.",
                tool_calls=[ToolCall("find_hotels",
                                     {"place": "Goa", "stars": 5, "nights": 5})],
            )

        # Turn 3 — first combo ($1,480 + $3,650 = $5,130) is over the $5,000 cap.
        #          Relax dates ±3d and search flights again.
        if flight_searches < 2:
            return Decision(
                thought=("Cheapest Qatar $1,480 + cheapest 5★ ITC $3,650 = $5,130 — "
                         "over the $5,000 cap by $130. Relax dates ±3d and search again."),
                tool_calls=[ToolCall("search_flights",
                                     {"from_": "BLR", "to": "GOI", "dates": "±3d"})],
            )

        # Turn 4 — cheaper combo fits ($1,090 + $3,650 = $4,740). Book & save. Done.
        return Decision(
            thought=("Qatar $1,090 + ITC $3,650 = $4,740 ≤ $5,000. It fits. "
                     "Book the flight and hotel, and save the preference."),
            tool_calls=[
                ToolCall("book_flight", {"flight": "Qatar"}),
                ToolCall("book_hotel", {"hotel": "ITC Grand Goa"}),
                ToolCall("save_user_profile", {"prefers_stay": "5★ Goa"}),
            ],
            final="Booked: Qatar (±3d) + ITC Grand Goa · 5 nights · $4,740 — fits the budget.",
        )
