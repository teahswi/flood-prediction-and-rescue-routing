"""
Feature engineering for flood risk prediction.
Combines rainfall, terrain, and historical data into normalized feature vectors.

Flood risk formula:
    R = w_r * R_r + w_t * R_t + w_h * R_h + w_d * R_d

Where:
    R_r: rainfall-related risk (current + antecedent + forecast)
    R_t: terrain susceptibility (elevation, slope, depressions)
    R_h: historical flood susceptibility
    R_d: drainage-related risk
    w_*: documented weights from config

NOTE: This produces a risk score [0-1], NOT a calibrated probability.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime


@dataclass
class RainfallFeatures:
    """Rainfall-related features used for flood risk estimation."""
    current_mm_hr: float
    last_24h_mm: float
    last_72h_mm: float        # antecedent rainfall
    forecast_1h_mm: float
    forecast_3h_mm: float
    forecast_6h_mm: float
    imd_alert_level: str      # e.g. "Green", "Yellow", "Orange", "Red"

    def compute_risk_score(self) -> float:
        """
        Returns rainfall risk in [0, 1].

        Components:
          - current_norm  : current_mm_hr normalised by 115.6 (IMD extremely heavy threshold)
          - antecedent_norm: last_72h_mm normalised by 300 mm (soil saturation threshold)
          - forecast_norm : forecast_3h_mm normalised by 50 mm
          - Weighted sum  : 0.5*current + 0.3*antecedent + 0.2*forecast
        """
        current_norm = float(np.clip(self.current_mm_hr / 115.6, 0.0, 1.0))
        antecedent_norm = float(np.clip(self.last_72h_mm / 300.0, 0.0, 1.0))
        forecast_norm = float(np.clip(self.forecast_3h_mm / 50.0, 0.0, 1.0))
        return 0.5 * current_norm + 0.3 * antecedent_norm + 0.2 * forecast_norm


@dataclass
class TerrainFeatures:
    """Terrain-related features used for flood risk estimation."""
    elevation_m: float
    slope_deg: float
    is_depression: bool
    flow_accumulation: float
    distance_to_water_m: float = 1000.0   # estimated distance to nearest water body

    def compute_risk_score(self, region_max_elevation: float = 50.0) -> float:
        """
        Returns terrain risk in [0, 1].

        Components:
          - elevation_risk  : low-lying areas score higher (inverted normalised elevation)
          - slope_risk      : flat areas retain water; steep slopes drain quickly
          - depression_bonus: 0.2 bonus if the cell is a topographic depression
          - flow_risk       : high flow accumulation indicates convergence of runoff
          - Result clipped to [0, 1]
        """
        elevation_risk = 1.0 - float(np.clip(self.elevation_m / region_max_elevation, 0.0, 1.0))

        if self.slope_deg >= 15.0:
            slope_risk = 0.0          # water drains away rapidly on steep slopes
        else:
            slope_risk = (15.0 - self.slope_deg) / 15.0

        depression_bonus = 0.2 if self.is_depression else 0.0

        flow_risk = float(np.clip(self.flow_accumulation / 100.0, 0.0, 1.0)) * 0.4

        raw = 0.3 * elevation_risk + 0.2 * slope_risk + depression_bonus + flow_risk
        return float(np.clip(raw, 0.0, 1.0))


@dataclass
class HistoricalFeatures:
    """Historical flood event data for a location."""
    flood_events_count: int = 0               # number of known historical events
    last_flood_year: Optional[int] = None
    historical_risk_score: float = 0.0        # pre-computed, normalised [0-1]

    def compute_risk_score(self) -> float:
        """
        Returns historical risk in [0, 1].

        If no historical data is available, returns 0.3 (neutral/unknown default).
        Otherwise returns the pre-computed historical_risk_score.
        """
        if self.flood_events_count == 0 and self.last_flood_year is None and self.historical_risk_score == 0.0:
            return 0.3   # neutral/unknown
        return float(np.clip(self.historical_risk_score, 0.0, 1.0))


@dataclass
class FloodFeatureVector:
    """
    Composite feature vector for a single spatial point and time.

    Aggregates rainfall, terrain, and historical sub-features and
    computes a weighted composite risk score.
    """
    lat: float
    lon: float
    timestamp: datetime
    rainfall: RainfallFeatures
    terrain: TerrainFeatures
    historical: HistoricalFeatures
    data_quality: Dict[str, bool] = field(default_factory=dict)
    # e.g. {"rainfall": True, "terrain": True, "historical": False, "drainage": False}

    def compute_composite_risk(self, weights: dict = None) -> float:
        """
        Computes the weighted composite flood risk score R in [0, 1].

        Default weights:
            rainfall   : 0.35
            terrain    : 0.30
            historical : 0.20
            drainage   : 0.15  (placeholder; returns 0 when drainage data absent)

        Formula:
            R = w_r*R_r + w_t*R_t + w_h*R_h + w_d*R_d
        """
        if weights is None:
            weights = {
                "rainfall": 0.35,
                "terrain": 0.30,
                "historical": 0.20,
                "drainage": 0.15,
            }

        r_rainfall = self.rainfall.compute_risk_score()
        r_terrain = self.terrain.compute_risk_score()
        r_historical = self.historical.compute_risk_score()
        drainage_risk = 0.0   # placeholder; no drainage model connected yet

        composite = (
            weights["rainfall"] * r_rainfall
            + weights["terrain"] * r_terrain
            + weights["historical"] * r_historical
            + weights["drainage"] * drainage_risk
        )
        return float(np.clip(composite, 0.0, 1.0))

    def missing_inputs(self) -> List[str]:
        """
        Returns a list of data sources that are missing or unavailable,
        based on the data_quality flags.

        A source is considered missing when its flag is False or it is
        absent from the data_quality dict.
        """
        all_sources = ["rainfall", "terrain", "historical", "drainage"]
        missing = []
        for source in all_sources:
            if not self.data_quality.get(source, False):
                missing.append(source)
        return missing

    def to_dict(self) -> dict:
        """Returns a flat dictionary suitable for logging or CSV export."""
        return {
            "lat": self.lat,
            "lon": self.lon,
            "timestamp": self.timestamp.isoformat(),
            # Rainfall
            "rainfall_current_mm_hr": self.rainfall.current_mm_hr,
            "rainfall_last_24h_mm": self.rainfall.last_24h_mm,
            "rainfall_last_72h_mm": self.rainfall.last_72h_mm,
            "rainfall_forecast_1h_mm": self.rainfall.forecast_1h_mm,
            "rainfall_forecast_3h_mm": self.rainfall.forecast_3h_mm,
            "rainfall_forecast_6h_mm": self.rainfall.forecast_6h_mm,
            "rainfall_imd_alert_level": self.rainfall.imd_alert_level,
            "rainfall_risk_score": self.rainfall.compute_risk_score(),
            # Terrain
            "terrain_elevation_m": self.terrain.elevation_m,
            "terrain_slope_deg": self.terrain.slope_deg,
            "terrain_is_depression": self.terrain.is_depression,
            "terrain_flow_accumulation": self.terrain.flow_accumulation,
            "terrain_distance_to_water_m": self.terrain.distance_to_water_m,
            "terrain_risk_score": self.terrain.compute_risk_score(),
            # Historical
            "historical_flood_events_count": self.historical.flood_events_count,
            "historical_last_flood_year": self.historical.last_flood_year,
            "historical_risk_score_raw": self.historical.historical_risk_score,
            "historical_risk_score": self.historical.compute_risk_score(),
            # Composite
            "composite_risk_score": self.compute_composite_risk(),
            # Quality
            "data_quality": str(self.data_quality),
            "missing_inputs": str(self.missing_inputs()),
        }


def build_feature_vector(
    lat: float,
    lon: float,
    weather_record=None,
    terrain_point=None,
    historical_data: dict = None,
) -> FloodFeatureVector:
    """
    Assembles a FloodFeatureVector from loosely-typed data objects.

    Parameters
    ----------
    lat, lon        : Geographic coordinates.
    weather_record  : Object or dict with rainfall attributes, or None.
    terrain_point   : Object or dict with terrain attributes, or None.
    historical_data : Dict with historical attributes, or None.

    All missing inputs are replaced with safe defaults and marked as
    unavailable in data_quality.
    """
    data_quality: Dict[str, bool] = {}

    # ------------------------------------------------------------------ #
    # Rainfall
    # ------------------------------------------------------------------ #
    def _get(obj, key, default):
        """Safely get a value from a dict or object attribute."""
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    if weather_record is not None:
        rainfall = RainfallFeatures(
            current_mm_hr=float(_get(weather_record, "current_mm_hr", 0.0)),
            last_24h_mm=float(_get(weather_record, "last_24h_mm", 0.0)),
            last_72h_mm=float(_get(weather_record, "last_72h_mm", 0.0)),
            forecast_1h_mm=float(_get(weather_record, "forecast_1h_mm", 0.0)),
            forecast_3h_mm=float(_get(weather_record, "forecast_3h_mm", 0.0)),
            forecast_6h_mm=float(_get(weather_record, "forecast_6h_mm", 0.0)),
            imd_alert_level=str(_get(weather_record, "imd_alert_level", "Green")),
        )
        data_quality["rainfall"] = True
    else:
        rainfall = RainfallFeatures(
            current_mm_hr=0.0,
            last_24h_mm=0.0,
            last_72h_mm=0.0,
            forecast_1h_mm=0.0,
            forecast_3h_mm=0.0,
            forecast_6h_mm=0.0,
            imd_alert_level="Green",
        )
        data_quality["rainfall"] = False

    # ------------------------------------------------------------------ #
    # Terrain
    # ------------------------------------------------------------------ #
    if terrain_point is not None:
        terrain = TerrainFeatures(
            elevation_m=float(_get(terrain_point, "elevation_m", 10.0)),
            slope_deg=float(_get(terrain_point, "slope_deg", 0.0)),
            is_depression=bool(_get(terrain_point, "is_depression", False)),
            flow_accumulation=float(_get(terrain_point, "flow_accumulation", 0.0)),
            distance_to_water_m=float(_get(terrain_point, "distance_to_water_m", 1000.0)),
        )
        data_quality["terrain"] = True
    else:
        terrain = TerrainFeatures(
            elevation_m=10.0,
            slope_deg=0.0,
            is_depression=False,
            flow_accumulation=0.0,
            distance_to_water_m=1000.0,
        )
        data_quality["terrain"] = False

    # ------------------------------------------------------------------ #
    # Historical
    # ------------------------------------------------------------------ #
    if historical_data is not None:
        historical = HistoricalFeatures(
            flood_events_count=int(_get(historical_data, "flood_events_count", 0)),
            last_flood_year=_get(historical_data, "last_flood_year", None),
            historical_risk_score=float(_get(historical_data, "historical_risk_score", 0.0)),
        )
        data_quality["historical"] = True
    else:
        historical = HistoricalFeatures()
        data_quality["historical"] = False

    # Drainage data is not yet integrated
    data_quality["drainage"] = False

    return FloodFeatureVector(
        lat=lat,
        lon=lon,
        timestamp=datetime.utcnow(),
        rainfall=rainfall,
        terrain=terrain,
        historical=historical,
        data_quality=data_quality,
    )
