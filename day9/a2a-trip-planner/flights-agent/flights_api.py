# ---------------------------------------------------------------------------
# flights_api.py  ·  the Flights Agent's real TOOL
# ---------------------------------------------------------------------------
# Flight fare/schedule data is licensed (GDS), so every real provider needs a
# key. To keep the demo bulletproof in a classroom, this tool NEVER dead-ends:
# if no key is set it returns route-aware SAMPLE data (clearly labelled), so
# the A2A + LLM flow always has something to show.
#
# FLIGHTS_PROVIDER picks the live provider when its key is present:
#   duffel        real REST flight SEARCH, free 2-min signup. Test-mode data is
#                 Duffel Airways sandbox (not real fares); live mode = real.
#   aviationstack live flight STATUS on a route right now (no search/dates).
#   amadeus       (being decommissioned 2026-07-17 — legacy keys only).
# No usable key  -> _sample() (source="sample").  FLIGHTS_STUB=1 forces sample.
# ---------------------------------------------------------------------------

import os
import zlib
import random
import datetime
import httpx

FLIGHTS_PROVIDER = os.getenv("FLIGHTS_PROVIDER", "duffel").lower()


def _tomorrow():
    return (datetime.date.today() + datetime.timedelta(days=1)).isoformat()


def _hm(iso):
    return iso[11:16] if iso and len(iso) >= 16 else ""


def _plus(dep_iso, arr_iso):
    return "+1" if (dep_iso and arr_iso and arr_iso[:10] > dep_iso[:10]) else ""


import math

# ---- graceful, route-aware SAMPLE data (keyless, always works) -------------
_AIRLINES = [("Delta", "DL"), ("United", "UA"), ("American", "AA"),
             ("JetBlue", "B6"), ("British Airways", "BA"), ("Lufthansa", "LH"),
             ("Air France", "AF"), ("Emirates", "EK")]

# a few dozen major airports so sample durations/prices look realistic per route
_AIRPORTS = {
    "ATL": (33.64, -84.43), "LAX": (33.94, -118.41), "ORD": (41.98, -87.90),
    "DFW": (32.90, -97.04), "DEN": (39.86, -104.67), "JFK": (40.64, -73.78),
    "EWR": (40.69, -74.17), "LGA": (40.78, -73.87), "SFO": (37.62, -122.38),
    "SEA": (47.45, -122.31), "LAS": (36.08, -115.15), "MCO": (28.43, -81.31),
    "MIA": (25.79, -80.29), "TPA": (27.98, -82.53), "BOS": (42.36, -71.01),
    "PHX": (33.43, -112.01), "IAH": (29.98, -95.34), "MSP": (44.88, -93.22),
    "DTW": (42.21, -83.35), "PHL": (39.87, -75.24), "IAD": (38.95, -77.46),
    "DCA": (38.85, -77.04), "SAN": (32.73, -117.19), "AUS": (30.19, -97.67),
    "BWI": (39.18, -76.67), "LHR": (51.47, -0.45), "LGW": (51.15, -0.19),
    "CDG": (49.01, 2.55), "AMS": (52.31, 4.76), "FRA": (50.04, 8.56),
    "MAD": (40.47, -3.56), "BCN": (41.30, 2.08), "FCO": (41.80, 12.24),
    "DUB": (53.43, -6.24), "ZRH": (47.46, 8.55), "IST": (41.28, 28.75),
    "DXB": (25.25, 55.36), "DOH": (25.27, 51.61), "SIN": (1.36, 103.99),
    "HKG": (22.31, 113.91), "NRT": (35.77, 140.39), "ICN": (37.46, 126.44),
    "SYD": (-33.95, 151.18), "YYZ": (43.68, -79.61), "YVR": (49.19, -123.18),
    "GRU": (-23.43, -46.47), "MEX": (19.44, -99.07), "DEL": (28.56, 77.10),
    "BOM": (19.09, 72.87), "MAA": (12.99, 80.17),
}


def _dist_km(a, b):
    if a not in _AIRPORTS or b not in _AIRPORTS:
        return 1200.0                                  # sensible default for unknown pairs
    (la1, lo1), (la2, lo2) = _AIRPORTS[a], _AIRPORTS[b]
    dlat, dlon = math.radians(la2 - la1), math.radians(lo2 - lo1)
    h = math.sin(dlat / 2) ** 2 + math.cos(math.radians(la1)) * math.cos(math.radians(la2)) * math.sin(dlon / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def _sample(origin, destination, date, note=None):
    # deterministic per route: same origin/destination -> same flights each run
    seed = zlib.crc32((origin + ">" + destination).encode())
    rnd = random.Random(seed)
    dist = _dist_km(origin, destination)
    nonstop_min = int(dist / 800 * 60) + 40            # ~800 km/h cruise + taxi/climb
    picks = rnd.sample(_AIRLINES, 3)
    flights = []
    for i, (name, code) in enumerate(picks):
        dep_h = rnd.choice([6, 8, 10, 13, 16, 18, 21, 22])
        dep_m = rnd.choice([5, 15, 30, 45, 55])
        stops = 0 if i == 0 else rnd.choice([0, 1])    # first is always nonstop
        dur = nonstop_min + rnd.randint(-10, 20) + stops * rnd.randint(75, 150)
        dep_abs = dep_h * 60 + dep_m
        arr_abs = dep_abs + dur
        plus = "+1" if arr_abs >= 1440 else ""
        arr_abs %= 1440
        price = 45 + int(dist * rnd.uniform(0.09, 0.15)) + stops * 40   # ~ economy $/km
        flights.append({
            "airline": name, "number": f"{code}{rnd.randint(100, 1998)}",
            "dep_time": f"{dep_h:02d}:{dep_m:02d}",
            "arr_time": f"{arr_abs // 60:02d}:{arr_abs % 60:02d}{plus}",
            "stops": stops, "status": "nonstop" if stops == 0 else f"{stops} stop",
            "price": str(price), "currency": "USD",
        })
    return {"ok": True, "dep": origin, "arr": destination, "date": date,
            "count": len(flights), "flights": flights, "source": "sample", "note": note}


# ---- dispatch --------------------------------------------------------------
async def search_flights(origin: str, destination: str, departure_date: str = "") -> dict:
    origin, destination = (origin or "").upper(), (destination or "").upper()
    date = departure_date or _tomorrow()

    if os.getenv("FLIGHTS_STUB"):
        return _sample(origin, destination, date)

    if FLIGHTS_PROVIDER == "duffel" and os.getenv("DUFFEL_TOKEN"):
        return await _duffel(origin, destination, date)
    if FLIGHTS_PROVIDER == "aviationstack" and os.getenv("AVIATIONSTACK_KEY"):
        return await _aviationstack(origin, destination)
    if FLIGHTS_PROVIDER == "amadeus" and os.getenv("AMADEUS_KEY") and os.getenv("AMADEUS_SECRET"):
        return await _amadeus(origin, destination, date)

    # no usable key -> keep the demo alive with clearly-labelled sample data
    return _sample(origin, destination, date,
                   note=f"no {FLIGHTS_PROVIDER} key set — showing sample data")


# ---- Duffel: real flight SEARCH (offer request -> offers) ------------------
async def _duffel(origin, destination, date) -> dict:
    token = os.getenv("DUFFEL_TOKEN")
    base = os.getenv("DUFFEL_BASE", "https://api.duffel.com")
    headers = {"Authorization": f"Bearer {token}",
               "Duffel-Version": os.getenv("DUFFEL_VERSION", "v2"),
               "Accept": "application/json", "Content-Type": "application/json"}
    body = {"data": {"slices": [{"origin": origin, "destination": destination, "departure_date": date}],
                     "passengers": [{"type": "adult"}], "cabin_class": "economy"}}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(base + "/air/offer_requests?return_offers=true",
                                  json=body, headers=headers)
        payload = r.json()
        if payload.get("errors"):
            e = payload["errors"][0]
            return {"ok": False, "error": e.get("message") or e.get("title", "duffel error")}
        offers = (payload.get("data") or {}).get("offers", [])
        flights = []
        for off in offers[:6]:
            segs = off["slices"][0]["segments"]
            first, last = segs[0], segs[-1]
            mk = first.get("marketing_carrier") or {}
            num = (mk.get("iata_code", "") +
                   (first.get("marketing_carrier_flight_number") or ""))
            flights.append({
                "airline": mk.get("name") or (off.get("owner") or {}).get("name", "—"),
                "number": num or "—",
                "dep_time": _hm(first.get("departing_at")),
                "arr_time": _hm(last.get("arriving_at")) + _plus(first.get("departing_at"), last.get("arriving_at")),
                "stops": len(segs) - 1,
                "status": "nonstop" if len(segs) == 1 else f"{len(segs)-1} stop",
                "price": off.get("total_amount"), "currency": off.get("total_currency", "USD"),
            })
        if not flights:
            return {"ok": False, "error": f"no offers for {origin} → {destination} on {date}"}
        return {"ok": True, "dep": origin, "arr": destination, "date": date,
                "count": len(flights), "flights": flights, "source": "duffel"}
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
        return {"ok": False, "error": f"duffel search failed: {exc}"}


# ---- Amadeus: legacy (portal decommissioning 2026-07-17) -------------------
async def _amadeus(origin, destination, date) -> dict:
    key, secret = os.getenv("AMADEUS_KEY"), os.getenv("AMADEUS_SECRET")
    base = os.getenv("AMADEUS_BASE", "https://test.api.amadeus.com")
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            tok = await client.post(base + "/v1/security/oauth2/token",
                data={"grant_type": "client_credentials", "client_id": key, "client_secret": secret},
                headers={"Content-Type": "application/x-www-form-urlencoded"})
            access = tok.json().get("access_token")
            if not access:
                return {"ok": False, "error": "Amadeus auth failed"}
            r = await client.get(base + "/v2/shopping/flight-offers",
                params={"originLocationCode": origin, "destinationLocationCode": destination,
                        "departureDate": date, "adults": 1, "max": 6, "currencyCode": "USD"},
                headers={"Authorization": f"Bearer {access}"})
        payload = r.json()
        if "errors" in payload:
            e = payload["errors"][0]
            return {"ok": False, "error": f"{e.get('title','error')}: {e.get('detail','')}"}
        carriers = (payload.get("dictionaries") or {}).get("carriers", {})
        flights = []
        for off in payload.get("data", [])[:6]:
            segs = off["itineraries"][0]["segments"]
            first, last = segs[0], segs[-1]
            code = first["carrierCode"]
            flights.append({
                "airline": carriers.get(code, code).title(), "number": code + first["number"],
                "dep_time": _hm(first["departure"]["at"]),
                "arr_time": _hm(last["arrival"]["at"]) + _plus(first["departure"]["at"], last["arrival"]["at"]),
                "stops": len(segs) - 1, "status": "nonstop" if len(segs) == 1 else f"{len(segs)-1} stop",
                "price": off.get("price", {}).get("grandTotal"),
                "currency": off.get("price", {}).get("currency", "USD"),
            })
        if not flights:
            return {"ok": False, "error": f"no offers for {origin} → {destination}"}
        return {"ok": True, "dep": origin, "arr": destination, "date": date,
                "count": len(flights), "flights": flights, "source": "amadeus"}
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
        return {"ok": False, "error": f"flight search failed: {exc}"}


# ---- AviationStack: live status on a route (kept as an option) -------------
async def _aviationstack(dep, arr) -> dict:
    key = os.getenv("AVIATIONSTACK_KEY")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get("http://api.aviationstack.com/v1/flights",
                                 params={"access_key": key, "dep_iata": dep, "arr_iata": arr, "limit": 20})
        payload = r.json()
        if "error" in payload:
            return {"ok": False, "error": str(payload["error"].get("message", payload["error"]))}
        flights, seen = [], set()
        for f in payload.get("data", []):
            if (f.get("flight") or {}).get("codeshared"):
                continue
            dep_t = _hm((f.get("departure") or {}).get("scheduled"))
            arr_t = _hm((f.get("arrival") or {}).get("scheduled"))
            number = (f.get("flight") or {}).get("iata") or (f.get("flight") or {}).get("number", "—")
            if (number, dep_t, arr_t) in seen:
                continue
            seen.add((number, dep_t, arr_t))
            flights.append({"airline": (f.get("airline") or {}).get("name", "—"), "number": number,
                            "status": f.get("flight_status", "—"), "dep_time": dep_t, "arr_time": arr_t,
                            "stops": 0, "price": None, "currency": None})
            if len(flights) >= 6:
                break
        if not flights:
            return {"ok": False, "error": f"no live flights on {dep} → {arr} right now"}
        return {"ok": True, "dep": dep, "arr": arr, "count": len(flights), "flights": flights, "source": "aviationstack"}
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        return {"ok": False, "error": f"flight lookup failed: {exc}"}
