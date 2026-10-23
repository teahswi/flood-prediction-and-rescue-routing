"""
config.py — Central configuration for the Flood Prediction & Rescue Routing system.

All dataclasses expose typed, documented fields with sensible defaults.
Environment variables (loaded from .env) can override any AppConfig field.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env overrides (silent if file is absent)
# ---------------------------------------------------------------------------
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    """Top-level application settings."""

    study_area_name: str = "Mumbai, Maharashtra, India"
    # (latitude, longitude) of the area centre
    study_area_coords: tuple = (19.0760, 72.8777)
    study_area_radius_km: float = 15.0
    cache_dir: str = "./cache"
    log_level: str = "INFO"
    data_refresh_interval_minutes: int = 30

    def __post_init__(self) -> None:
        # Allow environment-variable overrides for the most commonly changed fields
        self.cache_dir = os.getenv("CACHE_DIR", self.cache_dir)
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)
        env = os.getenv("APP_ENV", "development")
        if env == "production":
            self.log_level = "WARNING"


@dataclass
class FloodRiskConfig:
    """
    Weights and thresholds used by the flood-risk scoring model.

    Risk score = weight_rainfall * R_norm
               + weight_terrain  * T_norm
               + weight_history  * H_norm
               + weight_drainage * D_norm
    where each *_norm component is in [0, 1].
    """

    # --- Composite score weights (must sum to 1.0) ---
    weight_rainfall: float = 0.35
    weight_terrain: float = 0.30
    weight_history: float = 0.20
    weight_drainage: float = 0.15

    # --- Risk-level thresholds (score in [0, 1]) ---
    risk_low_threshold: float = 0.30
    risk_medium_threshold: float = 0.60
    risk_high_threshold: float = 0.80

    # --- IMD rainfall intensity class boundaries (mm/hr) ---
    rainfall_low_mm_hr: float = 5.0            # < 5      → negligible
    rainfall_moderate_mm_hr: float = 15.0      # 5–15     → moderate
    rainfall_heavy_mm_hr: float = 35.5         # 15–35.5  → heavy
    rainfall_very_heavy_mm_hr: float = 64.5    # 35.5–64.5 → very heavy
    rainfall_extremely_heavy_mm_hr: float = 115.6  # 64.5–115.6 → extremely heavy
    # > 115.6 mm/hr → catastrophic

    # Hours of prior rainfall to include in the antecedent moisture calculation
    antecedent_hours: int = 72

    def __post_init__(self) -> None:
        total = (
            self.weight_rainfall
            + self.weight_terrain
            + self.weight_history
            + self.weight_drainage
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"FloodRiskConfig weights must sum to 1.0, got {total:.4f}"
            )


@dataclass
class RoutingConfig:
    """
    Parameters for the hazard-aware shortest-path routing algorithm.

    Edge cost function:
        cost(e) = travel_time(e)
                + λ * hazard_penalty(e)
                + μ * uncertainty_penalty(e)
    """

    # Cost-function multipliers
    hazard_weight_lambda: float = 2.0    # λ — scales the hazard penalty term
    uncertainty_weight_mu: float = 0.5  # μ — scales prediction-uncertainty penalty

    # Free-flow speeds by road class (km/h)
    speed_kmh_motorway: float = 80.0
    speed_kmh_primary: float = 50.0
    speed_kmh_secondary: float = 40.0
    speed_kmh_residential: float = 25.0
    speed_kmh_default: float = 30.0

    # Risk thresholds governing edge accessibility
    impassable_risk_threshold: float = 0.85   # edges above this are removed from graph
    high_hazard_risk_threshold: float = 0.65  # edges above this receive heavy penalty


@dataclass
class SimulationConfig:
    """
    Parameters for the simplified rainfall-runoff flood simulation.

    Uses the NRCS/SCS Curve-Number method for runoff estimation and a
    uniform-flow approximation for inundation depth propagation.
    """

    # Spatial resolution matching SRTM 1-arc-second DEM (~30 m at equator)
    grid_resolution_m: float = 30.0

    # Temporal discretisation
    time_step_minutes: int = 30
    max_simulation_hours: int = 12

    # Rational-method runoff coefficients (dimensionless, 0–1)
    runoff_coefficient_urban: float = 0.85
    runoff_coefficient_suburban: float = 0.65
    runoff_coefficient_rural: float = 0.35

    # SCS Curve Numbers (dimensionless, higher → more runoff)
    cn_urban: int = 90
    cn_suburban: int = 75
    cn_rural: int = 60

    # Steady-state soil infiltration capacity (Green-Ampt approximation)
    infiltration_rate_mm_hr: float = 5.0


# ---------------------------------------------------------------------------
# Module-level singleton instances
# ---------------------------------------------------------------------------
CONFIG = AppConfig()
FLOOD_CONFIG = FloodRiskConfig()
ROUTING_CONFIG = RoutingConfig()
SIM_CONFIG = SimulationConfig()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def get_risk_label(score: float) -> str:
    """
    Convert a continuous risk score in [0, 1] to a human-readable label.

    Thresholds are taken from :data:`FLOOD_CONFIG`.

    Parameters
    ----------
    score:
        Composite flood-risk score in the range [0, 1].

    Returns
    -------
    str
        One of ``"Low"``, ``"Moderate"``, ``"High"``, or ``"Critical"``.
    """
    if score < FLOOD_CONFIG.risk_low_threshold:
        return "Low"
    if score < FLOOD_CONFIG.risk_medium_threshold:
        return "Moderate"
    if score < FLOOD_CONFIG.risk_high_threshold:
        return "High"
    return "Critical"


def get_risk_color(score: float) -> str:
    """
    Return a hex colour string matching the risk level for map visualisation.

    Parameters
    ----------
    score:
        Composite flood-risk score in the range [0, 1].

    Returns
    -------
    str
        Hex colour:
        - ``"#2ecc71"`` — green  (Low)
        - ``"#f39c12"`` — orange (Moderate)
        - ``"#e74c3c"`` — red    (High)
        - ``"#8e44ad"`` — purple (Critical)
    """
    label = get_risk_label(score)
    return {
        "Low": "#2ecc71",
        "Moderate": "#f39c12",
        "High": "#e74c3c",
        "Critical": "#8e44ad",
    }[label]
