# ---------------------------------------------------------------------------
# status_api.py  ·  the Flight Status Agent's streaming TOOL
# ---------------------------------------------------------------------------
# Flight *status* changes over time, so this tool doesn't return once — it is an
# async generator that YIELDS updates as the flight moves through its phases.
# That is what makes message/stream (SSE) real: the agent narrates the work.
#
# Same keyless-first philosophy as the flights search tool: with no key set it
# walks a deterministic SAMPLE flight (clearly labelled) so the demo always has
# something live to show. STATUS_PROVIDER=aviationstack (with a key) fetches a
# real status snapshot instead.
#
#   STATUS_TICK   seconds between streamed updates (default 1.4 -> ~8s run)
#   STATUS_STUB=1 forces the sample stream even if a key is present
# ---------------------------------------------------------------------------

import os
import zlib
import asyncio
import httpx

STATUS_PROVIDER = os.getenv("STATUS_PROVIDER", "sample").lower()
TICK = float(os.getenv("STATUS_TICK", "1.4"))

_TERMINALS = ["A", "B", "C", "D"]


# ---- graceful, deterministic SAMPLE stream (keyless, always works) ---------
async def _sample_stream(flight_number: str):
    fn = (flight_number or "XX000").upper().replace(" ", "")
    seed = zlib.crc32(fn.encode())
    gate = f"{_TERMINALS[seed % 4]}{(seed // 4) % 40 + 1}"
    arrival_gate = f"{_TERMINALS[(seed // 7) % 4]}{(seed // 11) % 40 + 1}"
    dep = f"{6 + seed % 14:02d}:{((seed // 3) % 6) * 10:02d}"

    # (task lifecycle state, human phase, detail, progress %)
    phases = [
        ("working",   "Scheduled", f"On time \u00b7 departs gate {gate} at {dep}", 0),
        ("working",   "Boarding",  f"Boarding now at gate {gate}",                18),
        ("working",   "Departed",  "Pushed back \u00b7 wheels up",                42),
        ("working",   "En route",  "Cruising at 38,000 ft",                       68),
        ("working",   "Landing",   "On final approach",                           90),
        ("completed", "Landed",    f"Arrived \u00b7 taxiing to gate {arrival_gate}", 100),
    ]
    last = len(phases) - 1
    for i, (state, phase, detail, pct) in enumerate(phases):
        if i:
            await asyncio.sleep(TICK)
        yield {
            "state": state, "phase": phase, "detail": detail, "final": i == last,
            "flight": {"number": fn, "gate": gate, "arrival_gate": arrival_gate, "progress": pct},
            "source": "sample",
        }


# ---- dispatch --------------------------------------------------------------
async def stream_status(flight_number: str, date: str = ""):
    """Yield status updates for a flight. Default = keyless sample stream."""
    if not os.getenv("STATUS_STUB") and STATUS_PROVIDER == "aviationstack" and os.getenv("AVIATIONSTACK_KEY"):
        async for u in _aviationstack_stream(flight_number):
            yield u
        return
    async for u in _sample_stream(flight_number):
        yield u


# ---- AviationStack: one real status snapshot, emitted as two frames --------
async def _aviationstack_stream(flight_number: str):
    key = os.getenv("AVIATIONSTACK_KEY")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get("http://api.aviationstack.com/v1/flights",
                                 params={"access_key": key, "flight_iata": flight_number, "limit": 1})
        data = (r.json().get("data") or [])
        if not data:
            yield {"state": "completed", "phase": "Unknown", "final": True, "source": "aviationstack",
                   "detail": f"No live data for {flight_number}", "flight": {"number": flight_number}}
            return
        f = data[0]
        st = f.get("flight_status", "unknown")
        dep, arr = (f.get("departure") or {}), (f.get("arrival") or {})
        airline = (f.get("airline") or {}).get("name", "")
        yield {"state": "working", "phase": st.title(), "final": False, "source": "aviationstack",
               "detail": f"{airline} {flight_number} \u00b7 {st}",
               "flight": {"number": flight_number, "gate": dep.get("gate")}}
        await asyncio.sleep(TICK)
        yield {"state": "completed", "phase": st.title(), "final": True, "source": "aviationstack",
               "detail": f"Status: {st} \u00b7 arrival {arr.get('airport', '')}".strip(),
               "flight": {"number": flight_number, "arrival_gate": arr.get("gate"), "status": st}}
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        yield {"state": "completed", "phase": "Error", "final": True, "source": "aviationstack",
               "detail": f"status lookup failed: {exc}", "flight": {"number": flight_number}}
