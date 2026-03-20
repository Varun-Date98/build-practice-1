"""
Open-Meteo-backed weather lookup: geocoding, current conditions, forecast daily,
historical archive, and climate projections for far-future dates.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any

import httpx

# Routing uses UTC calendar dates (documented tradeoff vs city-local midnight).
FORECAST_HORIZON_DAYS = 16
CLIMATE_MAX_DATE = date(2050, 1, 1)
HISTORICAL_MIN_DATE = date(1940, 1, 1)
CLIMATE_MODEL = "EC_Earth3P_HR"

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
CLIMATE_URL = "https://climate-api.open-meteo.com/v1/climate"

# Shared where supported; climate uses a slightly different set (no weather_code).
DAILY_FORECAST_ARCHIVE = (
    "temperature_2m_max,temperature_2m_min,precipitation_sum,"
    "wind_speed_10m_max,wind_direction_10m_dominant,"
    "relative_humidity_2m_max,relative_humidity_2m_min,weather_code"
)
DAILY_CLIMATE = (
    "temperature_2m_max,temperature_2m_min,precipitation_sum,"
    "wind_speed_10m_max,relative_humidity_2m_mean,cloud_cover_mean"
)

CURRENT_VARS = (
    "temperature_2m,relative_humidity_2m,wind_speed_10m,"
    "wind_direction_10m,weather_code"
)

HTTP_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
CLIMATE_DISCLAIMER = (
    "Values come from a single downscaled CMIP6 climate model (not an operational "
    "weather forecast). Use for illustration only; uncertainty is large for a specific day."
)


class WeatherServiceError(Exception):
    """Raised for unexpected failures after HTTP success."""


def _err(message: str, detail: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"error": message}
    if detail:
        out["detail"] = detail
    return out


def _http_get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            r = client.get(url, params=params)
            r.raise_for_status()
    except httpx.TimeoutException:
        return _err("Weather request timed out", "The weather service did not respond in time.")
    except httpx.HTTPStatusError as e:
        body = ""
        try:
            body = e.response.text[:500]
        except Exception:
            pass
        return _err(
            "Weather service returned an HTTP error",
            f"status={e.response.status_code} body={body!r}",
        )
    except httpx.RequestError as e:
        return _err("Network error calling weather service", str(e))

    try:
        data = r.json()
    except json.JSONDecodeError as e:
        return _err("Invalid JSON from weather service", str(e))

    if data.get("error") is True:
        return _err(
            "Weather API error",
            str(data.get("reason", data)),
        )
    return data


def _parse_iso_date(s: str) -> date | dict[str, Any]:
    s = (s or "").strip()
    try:
        return date.fromisoformat(s)
    except ValueError:
        return _err(
            "Invalid date format",
            f"Expected YYYY-MM-DD, got {s!r}.",
        )


def geocode_city(city: str) -> dict[str, Any]:
    city = (city or "").strip()
    if not city:
        return _err("Missing city", "Provide a non-empty city name.")

    data = _http_get_json(
        GEOCODE_URL,
        {"name": city, "count": 5, "language": "en"},
    )
    if "error" in data:
        return data

    results = data.get("results") or []
    if not results:
        return _err(
            "No location found",
            f"No geocoding results for {city!r}. Try adding region or country.",
        )

    top = results[0]
    resolved = {
        "name": top.get("name"),
        "admin1": top.get("admin1"),
        "country": top.get("country"),
        "country_code": top.get("country_code"),
        "latitude": top.get("latitude"),
        "longitude": top.get("longitude"),
    }
    alternatives = [
        {
            "name": r.get("name"),
            "admin1": r.get("admin1"),
            "country": r.get("country"),
        }
        for r in results[1:5]
    ]
    return {"resolved": resolved, "alternatives": alternatives}


def _slice_first_daily(api_body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    daily = api_body.get("daily") or {}
    times = daily.get("time") or []
    if not times:
        raise WeatherServiceError("Missing daily.time in API response")
    row: dict[str, Any] = {"date": times[0]}
    for key, values in daily.items():
        if key == "time":
            continue
        if isinstance(values, list) and values:
            row[key] = values[0]
    units = api_body.get("daily_units") or {}
    return row, units


def _slice_current(api_body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    cur = api_body.get("current") or {}
    if not cur:
        raise WeatherServiceError("Missing current conditions in API response")
    units = api_body.get("current_units") or {}
    # Flatten to serializable dict (drop internal interval fields if present).
    out = {k: v for k, v in cur.items() if not str(k).endswith("_interval")}
    return out, units


def fetch_current(lat: float, lon: float) -> dict[str, Any]:
    data = _http_get_json(
        FORECAST_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "timezone": "auto",
            "current": CURRENT_VARS,
        },
    )
    if "error" in data:
        return data
    try:
        current, units = _slice_current(data)
    except WeatherServiceError as e:
        return _err("Unexpected forecast payload", str(e))
    return {
        "source": "current",
        "current": current,
        "current_units": units,
        "daily": None,
        "daily_units": None,
    }


def fetch_forecast_daily(lat: float, lon: float, d: date) -> dict[str, Any]:
    ds = d.isoformat()
    data = _http_get_json(
        FORECAST_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "timezone": "auto",
            "forecast_days": FORECAST_HORIZON_DAYS,
            "daily": DAILY_FORECAST_ARCHIVE,
            "start_date": ds,
            "end_date": ds,
        },
    )
    if "error" in data:
        return data
    try:
        daily_row, daily_units = _slice_first_daily(data)
    except WeatherServiceError as e:
        return _err("Unexpected forecast daily payload", str(e))
    return {
        "source": "forecast_daily",
        "current": None,
        "current_units": None,
        "daily": daily_row,
        "daily_units": daily_units,
    }


def fetch_historical_daily(lat: float, lon: float, d: date) -> dict[str, Any]:
    if d < HISTORICAL_MIN_DATE:
        return _err(
            "Date too far in the past",
            f"Historical data is not available before {HISTORICAL_MIN_DATE.isoformat()}.",
        )
    ds = d.isoformat()
    data = _http_get_json(
        ARCHIVE_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "timezone": "auto",
            "daily": DAILY_FORECAST_ARCHIVE,
            "start_date": ds,
            "end_date": ds,
        },
    )
    if "error" in data:
        return data
    try:
        daily_row, daily_units = _slice_first_daily(data)
    except WeatherServiceError as e:
        return _err("Unexpected historical payload", str(e))
    return {
        "source": "historical",
        "current": None,
        "current_units": None,
        "daily": daily_row,
        "daily_units": daily_units,
    }


def fetch_climate_daily(lat: float, lon: float, d: date) -> dict[str, Any]:
    if d > CLIMATE_MAX_DATE:
        return _err(
            "Date outside supported range",
            f"Climate projections are not available after {CLIMATE_MAX_DATE.isoformat()}.",
        )
    ds = d.isoformat()
    data = _http_get_json(
        CLIMATE_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "models": CLIMATE_MODEL,
            "daily": DAILY_CLIMATE,
            "start_date": ds,
            "end_date": ds,
        },
    )
    if "error" in data:
        return data
    try:
        daily_row, daily_units = _slice_first_daily(data)
    except WeatherServiceError as e:
        return _err("Unexpected climate API payload", str(e))
    return {
        "source": "climate_projection",
        "disclaimer": CLIMATE_DISCLAIMER,
        "current": None,
        "current_units": None,
        "daily": daily_row,
        "daily_units": daily_units,
    }


def fetch_weather_for_date(lat: float, lon: float, d: date) -> dict[str, Any]:
    """Route a calendar day to historical, forecast daily, or climate projection."""
    today_utc = date.today()
    if d < today_utc:
        return fetch_historical_daily(lat, lon, d)
    if d <= today_utc + timedelta(days=FORECAST_HORIZON_DAYS):
        return fetch_forecast_daily(lat, lon, d)
    if d <= CLIMATE_MAX_DATE:
        return fetch_climate_daily(lat, lon, d)
    return _err(
        "Date outside supported range",
        f"No data path for {d.isoformat()} (after {CLIMATE_MAX_DATE.isoformat()}).",
    )


def get_weather(city: str, date_str: str | None = None) -> dict[str, Any]:
    """
    Public tool entry: resolve city and return weather JSON for optional ISO date.
    """
    geo = geocode_city(city)
    if "error" in geo:
        return geo

    resolved = geo["resolved"]
    lat, lon = resolved["latitude"], resolved["longitude"]
    if lat is None or lon is None:
        return _err("Geocoding missing coordinates", json.dumps(resolved))

    base: dict[str, Any] = {
        "resolved_location": resolved,
        "alternatives_considered": geo.get("alternatives") or [],
    }

    if not date_str or not str(date_str).strip():
        wx = fetch_current(float(lat), float(lon))
        if "error" in wx:
            return {**base, **wx}
        out = {**base, **wx}
        out["date"] = None
        return out

    parsed = _parse_iso_date(date_str)
    if isinstance(parsed, dict):
        return {**base, **parsed}

    wx = fetch_weather_for_date(float(lat), float(lon), parsed)
    if "error" in wx:
        return {**base, **wx}
    out = {**base, **wx}
    out["date"] = parsed.isoformat()
    return out
