"""Online extras: weather (Open-Meteo) and encyclopedia lookups (English Wikipedia).

Both are free and need no key. Requests run in a thread, so listening never waits for them;
without internet the local brain says so (weather) or lets the LLM answer (Wikipedia).
"""

import asyncio
import json
import logging
import urllib.parse
import urllib.request

log = logging.getLogger(__name__)

AGENT = "NeuroVision-local-robot/1.0"
TIMEOUT = 8

# WMO weather codes (Open-Meteo)
WEATHER_CODES = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "foggy", 48: "foggy with rime",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "heavy showers", 85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail", 99: "severe thunderstorms with hail",
}


def _get(url: str, params: dict) -> dict:
    req = urllib.request.Request(f"{url}?{urllib.parse.urlencode(params)}", headers={"User-Agent": AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.load(r)


def _weather(place: str, days: int) -> dict:
    found = _get("https://geocoding-api.open-meteo.com/v1/search",
                 {"name": place, "count": 1, "language": "en", "format": "json"}).get("results")
    if not found:
        return {"error": f"No place called '{place}'."}
    loc = found[0]
    w = _get("https://api.open-meteo.com/v1/forecast", {
        "latitude": loc["latitude"], "longitude": loc["longitude"], "timezone": "auto",
        "forecast_days": days,
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
    })
    cur, d = w["current"], w["daily"]
    return {
        "place": ", ".join(x for x in (loc.get("name"), loc.get("admin1"), loc.get("country")) if x),
        "now": {
            "time": cur["time"][-5:], "temp_c": cur["temperature_2m"], "feels_like_c": cur["apparent_temperature"],
            "humidity": cur["relative_humidity_2m"], "condition": WEATHER_CODES.get(cur["weather_code"], "unknown"),
            "wind_kmh": cur["wind_speed_10m"],
        },
        "days": [{
            "date": d["time"][i], "condition": WEATHER_CODES.get(d["weather_code"][i], "unknown"),
            "min_c": d["temperature_2m_min"][i], "max_c": d["temperature_2m_max"][i],
            "rain_chance": d["precipitation_probability_max"][i],
        } for i in range(len(d["time"]))],
    }


def _wiki(query: str) -> dict:
    api = "https://en.wikipedia.org/w/api.php"
    hits = _get(api, {"action": "query", "list": "search", "srsearch": query, "srlimit": 1,
                      "format": "json"})["query"]["search"]
    if not hits:
        return {"results": []}
    pages = _get(api, {"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
                       "exchars": 700, "titles": hits[0]["title"], "format": "json"})["query"]["pages"].values()
    return {"source": "Wikipedia", "results": [
        {"title": p["title"], "summary": p.get("extract", "").strip()} for p in pages if p.get("extract")]}


async def weather(place: str, days: int = 3) -> dict:
    return await asyncio.to_thread(_weather, place, max(1, min(int(days), 7)))


async def wiki(query: str) -> dict:
    return await asyncio.to_thread(_wiki, query)
