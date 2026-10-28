"""Internetten bilgi: hava durumu (Open-Meteo) ve ansiklopedik arama (Vikipedi).

Ikisi de anahtarsiz ve ucretsiz; Gemini'nin Google aramasi bu API anahtarinda kotasiz.
Istekler is parcaciginda calisir, ses akisini bekletmez.
"""

import asyncio
import json
import logging
import urllib.parse
import urllib.request

log = logging.getLogger(__name__)

AGENT = "NeuroVision-robot/1.0 (https://github.com/drpeker/neurovision-gemini)"
TIMEOUT = 8

# WMO hava kodlari (Open-Meteo)
WEATHER_CODES = {
    0: "açık", 1: "çoğunlukla açık", 2: "parçalı bulutlu", 3: "kapalı",
    45: "sisli", 48: "kırağılı sis",
    51: "hafif çisenti", 53: "çisenti", 55: "yoğun çisenti", 56: "dondurucu çisenti", 57: "dondurucu çisenti",
    61: "hafif yağmur", 63: "yağmur", 65: "kuvvetli yağmur", 66: "dondurucu yağmur", 67: "dondurucu yağmur",
    71: "hafif kar", 73: "kar", 75: "yoğun kar", 77: "kar taneleri",
    80: "hafif sağanak", 81: "sağanak", 82: "şiddetli sağanak", 85: "kar sağanağı", 86: "yoğun kar sağanağı",
    95: "gök gürültülü fırtına", 96: "dolulu fırtına", 99: "şiddetli dolulu fırtına",
}


def _get(url: str, params: dict) -> dict:
    req = urllib.request.Request(f"{url}?{urllib.parse.urlencode(params)}", headers={"User-Agent": AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.load(r)


def _weather(place: str, days: int) -> dict:
    found = _get("https://geocoding-api.open-meteo.com/v1/search",
                 {"name": place, "count": 1, "language": "tr", "format": "json"}).get("results")
    if not found:
        return {"error": f"'{place}' adında bir yer bulunamadı."}
    loc = found[0]
    w = _get("https://api.open-meteo.com/v1/forecast", {
        "latitude": loc["latitude"], "longitude": loc["longitude"], "timezone": "auto",
        "forecast_days": days,
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
    })
    cur, d = w["current"], w["daily"]
    return {
        "yer": ", ".join(x for x in (loc.get("name"), loc.get("admin1"), loc.get("country")) if x),
        "simdi": {
            "saat": cur["time"][-5:], "sicaklik_c": cur["temperature_2m"],
            "hissedilen_c": cur["apparent_temperature"], "nem_yuzde": cur["relative_humidity_2m"],
            "durum": WEATHER_CODES.get(cur["weather_code"], "bilinmiyor"), "ruzgar_kmh": cur["wind_speed_10m"],
        },
        "gunler": [{
            "tarih": d["time"][i], "durum": WEATHER_CODES.get(d["weather_code"][i], "bilinmiyor"),
            "en_dusuk_c": d["temperature_2m_min"][i], "en_yuksek_c": d["temperature_2m_max"][i],
            "yagis_olasiligi_yuzde": d["precipitation_probability_max"][i],
        } for i in range(len(d["time"]))],
    }


def _wiki(query: str) -> dict:
    api = "https://tr.wikipedia.org/w/api.php"
    hits = _get(api, {"action": "query", "list": "search", "srsearch": query, "srlimit": 3,
                      "format": "json"})["query"]["search"]
    if not hits:
        return {"result": "Vikipedi'de sonuç bulunamadı."}
    pages = _get(api, {"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
                       "exchars": 700, "titles": "|".join(h["title"] for h in hits),
                       "format": "json"})["query"]["pages"].values()
    return {"kaynak": "Vikipedi", "sonuclar": [
        {"baslik": p["title"], "ozet": p.get("extract", "").strip()} for p in pages if p.get("extract")]}


async def weather(place: str, days: int = 3) -> dict:
    return await asyncio.to_thread(_weather, place, max(1, min(int(days), 7)))


async def wiki(query: str) -> dict:
    return await asyncio.to_thread(_wiki, query)
