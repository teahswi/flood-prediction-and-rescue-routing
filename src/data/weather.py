"""
Weather data ingestion from Open-Meteo API.
Open-Meteo is free and does not require an API key.
API docs: https://open-meteo.com/en/docs
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class WeatherRecord:
    """Holds weather data for a single location and time."""
    lat: float
    lon: float
    timestamp: datetime
    rainfall_mm_hr: float        # current hourly rainfall
    rainfall_24h_mm: float       # last 24h accumulated
    rainfall_72h_mm: float       # last 72h accumulated (antecedent)
    temperature_c: float
    humidity_pct: float
    wind_speed_ms: float
    forecast_1h_mm: float        # forecast next 1 hour
    forecast_3h_mm: float        # forecast next 3 hours
    forecast_6h_mm: float        # forecast next 6 hours
    source: str = "open-meteo"
    fetched_at: datetime = field(default_factory=datetime.utcnow)
    is_stale: bool = False


class WeatherClient:
    """
    Client for Open-Meteo weather API.
    Fetches current conditions, historical accumulations, and short-range forecasts.
    """

    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    HISTORICAL_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

    def __init__(self, timeout_seconds: int = 15):
        self.timeout = timeout_seconds
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    def _parse_hourly(self, data: dict) -> pd.DataFrame:
        """Parse the hourly block of an Open-Meteo JSON response into a DataFrame."""
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            return pd.DataFrame()

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(times),
            "precipitation_mm": hourly.get("precipitation", [np.nan] * len(times)),
            "temperature_c": hourly.get("temperature_2m", [np.nan] * len(times)),
            "humidity_pct": hourly.get("relative_humidity_2m", [np.nan] * len(times)),
            "wind_speed_ms": hourly.get("wind_speed_10m", [np.nan] * len(times)),
        })
        # Convert wind speed from km/h to m/s if the API returns km/h
        # Open-Meteo wind_speed_10m default unit is km/h
        df["wind_speed_ms"] = df["wind_speed_ms"] / 3.6
        return df.sort_values("timestamp").reset_index(drop=True)

    def fetch_current_and_forecast(self, lat: float, lon: float) -> WeatherRecord:
        """
        Fetch current weather and forecast for the given coordinates.

        Queries past 3 days and 2 forecast days, then derives:
        - rainfall_mm_hr: precipitation in the most recent completed hour
        - rainfall_24h_mm: sum of last 24 hourly values
        - rainfall_72h_mm: sum of last 72 hourly values
        - forecast_1h_mm / 3h / 6h: sums of upcoming hourly precipitation

        Returns a WeatherRecord; on failure returns a zero-filled stale record.
        """
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": "precipitation,temperature_2m,relative_humidity_2m,wind_speed_10m",
                "forecast_days": 2,
                "timezone": "UTC",
                "past_days": 3,
            }
            resp = self._session.get(self.BASE_URL, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            df = self._parse_hourly(data)
            if df.empty:
                raise ValueError("Empty hourly data returned from Open-Meteo.")

            now_utc = datetime.utcnow()
            # Ensure timezone-naive comparison
            df["timestamp"] = df["timestamp"].dt.tz_localize(None) if df["timestamp"].dt.tz is not None else df["timestamp"]

            past_mask = df["timestamp"] <= now_utc
            future_mask = df["timestamp"] > now_utc

            past_df = df[past_mask].copy()
            future_df = df[future_mask].copy()

            # Current hour rainfall
            rainfall_mm_hr = float(past_df["precipitation_mm"].iloc[-1]) if len(past_df) >= 1 else 0.0

            # 24h accumulation
            rainfall_24h_mm = float(past_df["precipitation_mm"].tail(24).sum()) if len(past_df) >= 1 else 0.0

            # 72h accumulation
            rainfall_72h_mm = float(past_df["precipitation_mm"].tail(72).sum()) if len(past_df) >= 1 else 0.0

            # Latest temperature, humidity, wind
            latest = past_df.iloc[-1] if not past_df.empty else df.iloc[0]
            temperature_c = float(latest["temperature_c"]) if not np.isnan(latest["temperature_c"]) else 0.0
            humidity_pct = float(latest["humidity_pct"]) if not np.isnan(latest["humidity_pct"]) else 0.0
            wind_speed_ms = float(latest["wind_speed_ms"]) if not np.isnan(latest["wind_speed_ms"]) else 0.0

            # Forecast accumulations
            forecast_1h_mm = float(future_df["precipitation_mm"].head(1).sum()) if not future_df.empty else 0.0
            forecast_3h_mm = float(future_df["precipitation_mm"].head(3).sum()) if not future_df.empty else 0.0
            forecast_6h_mm = float(future_df["precipitation_mm"].head(6).sum()) if not future_df.empty else 0.0

            return WeatherRecord(
                lat=lat,
                lon=lon,
                timestamp=now_utc,
                rainfall_mm_hr=max(rainfall_mm_hr, 0.0),
                rainfall_24h_mm=max(rainfall_24h_mm, 0.0),
                rainfall_72h_mm=max(rainfall_72h_mm, 0.0),
                temperature_c=temperature_c,
                humidity_pct=humidity_pct,
                wind_speed_ms=wind_speed_ms,
                forecast_1h_mm=max(forecast_1h_mm, 0.0),
                forecast_3h_mm=max(forecast_3h_mm, 0.0),
                forecast_6h_mm=max(forecast_6h_mm, 0.0),
                is_stale=False,
            )

        except Exception as exc:  # noqa: BLE001
            print(f"[WeatherClient] Failed to fetch weather for ({lat}, {lon}): {exc}")
            return WeatherRecord(
                lat=lat,
                lon=lon,
                timestamp=datetime.utcnow(),
                rainfall_mm_hr=0.0,
                rainfall_24h_mm=0.0,
                rainfall_72h_mm=0.0,
                temperature_c=0.0,
                humidity_pct=0.0,
                wind_speed_ms=0.0,
                forecast_1h_mm=0.0,
                forecast_3h_mm=0.0,
                forecast_6h_mm=0.0,
                is_stale=True,
            )

    def fetch_hourly_series(
        self, lat: float, lon: float, days_back: int = 7
    ) -> pd.DataFrame:
        """
        Fetch a historical hourly weather series.

        Args:
            lat: Latitude.
            lon: Longitude.
            days_back: Number of past days to retrieve (max 92 on free tier).

        Returns:
            DataFrame with columns:
            timestamp, precipitation_mm, temperature_c, humidity_pct, wind_speed_ms.
            Returns an empty DataFrame on error.
        """
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": "precipitation,temperature_2m,relative_humidity_2m,wind_speed_10m",
                "forecast_days": 1,
                "timezone": "UTC",
                "past_days": days_back,
            }
            resp = self._session.get(self.BASE_URL, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            df = self._parse_hourly(data)
            now_utc = datetime.utcnow()
            df["timestamp"] = df["timestamp"].dt.tz_localize(None) if df["timestamp"].dt.tz is not None else df["timestamp"]
            # Keep only past data
            df = df[df["timestamp"] <= now_utc].reset_index(drop=True)
            return df
        except Exception as exc:  # noqa: BLE001
            print(f"[WeatherClient] fetch_hourly_series failed for ({lat}, {lon}): {exc}")
            return pd.DataFrame(columns=["timestamp", "precipitation_mm", "temperature_c", "humidity_pct", "wind_speed_ms"])

    def get_imd_alert_level(self, rainfall_mm_hr: float) -> str:
        """
        Convert an hourly rainfall value (mm/hr) to an IMD color-coded alert level.

        IMD classification:
            <2.5       → No Alert
            2.5–7.5    → Green   (Light rain)
            7.5–35.5   → Yellow  (Moderate rain)
            35.5–64.5  → Orange  (Heavy rain)
            64.5–115.6 → Red     (Very heavy rain)
            >115.6     → Red+    (Extremely heavy)
        """
        if rainfall_mm_hr < 2.5:
            return "No Alert"
        elif rainfall_mm_hr < 7.5:
            return "Green"
        elif rainfall_mm_hr < 35.5:
            return "Yellow"
        elif rainfall_mm_hr < 64.5:
            return "Orange"
        elif rainfall_mm_hr <= 115.6:
            return "Red"
        else:
            return "Red+"


# ---------------------------------------------------------------------------
# Module-level convenience helpers
# ---------------------------------------------------------------------------

_default_client: Optional[WeatherClient] = None
_cache: dict = {}  # Simple in-process cache keyed by (lat_rounded, lon_rounded)
_CACHE_TTL_SECONDS = 1800  # 30 minutes


def _get_client() -> WeatherClient:
    global _default_client
    if _default_client is None:
        _default_client = WeatherClient()
    return _default_client


def get_weather(lat: float, lon: float, use_cache: bool = True) -> WeatherRecord:
    """
    Convenience function to fetch a WeatherRecord for (lat, lon).

    Results are cached in-process for 30 minutes (TTL=1800 s).  When
    ``use_cache=False`` the cache is bypassed and a fresh request is made.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        use_cache: Whether to return a cached result if available.

    Returns:
        WeatherRecord populated with current conditions and short-range forecast.
    """
    if not (-90.0 <= float(lat) <= 90.0):
        raise ValueError(f"Latitude must be between -90 and 90, got {lat}.")
    if not (-180.0 <= float(lon) <= 180.0):
        raise ValueError(f"Longitude must be between -180 and 180, got {lon}.")

    cache_key = f"weather_{float(lat):.3f}_{float(lon):.3f}"

    if use_cache and cache_key in _cache:
        record, cached_at = _cache[cache_key]
        age_seconds = (datetime.utcnow() - cached_at).total_seconds()
        if age_seconds < _CACHE_TTL_SECONDS:
            return record

    client = _get_client()
    record = client.fetch_current_and_forecast(lat, lon)

    if use_cache:
        _cache[cache_key] = (record, datetime.utcnow())

    return record
