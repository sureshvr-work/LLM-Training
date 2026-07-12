# ---------------------------------------------------------------------------
# weather_api.py  ·  the Weather Agent's real TOOL (Open-Meteo, no key)
# ---------------------------------------------------------------------------
# geocode the city -> lat/lon, then fetch the daily forecast. Returns a tidy
# dict the UI renders as a weather card. On any failure returns {ok: False,...}
# so the agent can still send a well-formed A2A response.
#
# WEATHER_STUB=1 returns canned data (offline dev / CI only — not a reasoner mock).
# ---------------------------------------------------------------------------

import os
import httpx

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather code -> (word, emoji) for the card.
_CODES = {
    0: ("clear", "\u2600\ufe0f"),
    1: ("mainly clear", "\U0001F324\ufe0f"), 2: ("partly cloudy", "\u26c5"), 3: ("overcast", "\u2601\ufe0f"),
    45: ("fog", "\U0001F32B\ufe0f"), 48: ("rime fog", "\U0001F32B\ufe0f"),
    51: ("light drizzle", "\U0001F327\ufe0f"), 53: ("drizzle", "\U0001F327\ufe0f"), 55: ("heavy drizzle", "\U0001F327\ufe0f"),
    61: ("light rain", "\U0001F327\ufe0f"), 63: ("rain", "\U0001F327\ufe0f"), 65: ("heavy rain", "\U0001F327\ufe0f"),
    71: ("light snow", "\U0001F328\ufe0f"), 73: ("snow", "\U0001F328\ufe0f"), 75: ("heavy snow", "\u2744\ufe0f"),
    80: ("showers", "\U0001F326\ufe0f"), 81: ("showers", "\U0001F326\ufe0f"), 82: ("violent showers", "\u26c8\ufe0f"),
    95: ("thunderstorm", "\u26c8\ufe0f"), 96: ("thunderstorm", "\u26c8\ufe0f"), 99: ("thunderstorm", "\u26c8\ufe0f"),
}


def _describe(code):
    return _CODES.get(code, ("mixed", "\U0001F324\ufe0f"))


def _stub(city, when):
    word, emoji = _describe(63)
    return {"ok": True, "city": city.title(), "country": "", "when": when,
            "date": "2026-07-11", "tmax": 19, "tmin": 12, "precip_prob": 70,
            "wind": 14, "code": 63, "condition": word, "emoji": emoji, "source": "stub"}


async def get_forecast(city: str, when: str = "tomorrow") -> dict:
    if os.getenv("WEATHER_STUB"):
        return _stub(city, when)

    index = 0 if str(when).strip().lower() == "today" else 1
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            geo = await client.get(GEOCODE_URL, params={"name": city, "count": 1, "language": "en"})
            results = geo.json().get("results")
            if not results:
                return {"ok": False, "error": f"could not find a place called '{city}'"}
            place = results[0]
            fc = await client.get(FORECAST_URL, params={
                "latitude": place["latitude"], "longitude": place["longitude"],
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,"
                         "weather_code,wind_speed_10m_max",
                "timezone": "auto", "forecast_days": 2,
            })
            daily = fc.json()["daily"]
        word, emoji = _describe(daily["weather_code"][index])
        return {
            "ok": True,
            "city": place["name"],
            "country": place.get("country", ""),
            "when": "today" if index == 0 else "tomorrow",
            "date": daily["time"][index],
            "tmax": round(daily["temperature_2m_max"][index]),
            "tmin": round(daily["temperature_2m_min"][index]),
            "precip_prob": daily["precipitation_probability_max"][index],
            "wind": round(daily["wind_speed_10m_max"][index]),
            "code": daily["weather_code"][index],
            "condition": word,
            "emoji": emoji,
            "source": "open-meteo",
        }
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
        return {"ok": False, "error": f"weather lookup failed: {exc}"}
