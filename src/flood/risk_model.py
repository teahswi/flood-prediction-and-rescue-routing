"""
Flood risk model: combines feature vectors into a final risk assessment.

Implements the weighted composite risk model:
    R = w_r*R_r + w_t*R_t + w_h*R_h + w_d*R_d

IMPORTANT LIMITATION: This model produces a risk score [0-1], not a
calibrated probability of flooding. The score is for relative comparison
and decision support only. Do not interpret score=0.8 as "80% chance of flooding"
without proper calibration against observed flood events.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime


@dataclass
class RiskAssessment:
    """
    Full risk assessment for a single spatial point.

    composite_risk_score is a risk index in [0, 1]; it is NOT a
    calibrated probability of flooding.
    """
    scenario_id: str
    lat: float
    lon: float
    timestamp: datetime
    composite_risk_score: float          # [0-1]
    risk_label: str                      # "Low", "Moderate", "High", "Critical"
    rainfall_risk: float
    terrain_risk: float
    historical_risk: float
    drainage_risk: float
    weights_used: Dict[str, float]
    data_gaps: List[str]
    model_version: str = "1.0.0-rule-based"
    uncertainty_note: str = "Score is a risk index, not a calibrated probability"
    computed_at: datetime = field(default_factory=datetime.utcnow)


class FloodRiskModel:
    """
    Weighted composite flood risk model.

    Combines rainfall, terrain, historical, and drainage risk sub-scores
    into a single composite risk index using configurable weights.

    Model version: 1.0.0-rule-based (deterministic, no ML inference).
    """

    MODEL_VERSION: str = "1.0.0-rule-based"

    _DEFAULT_WEIGHTS: Dict[str, float] = {
        "rainfall": 0.35,
        "terrain": 0.30,
        "historical": 0.20,
        "drainage": 0.15,
    }

    def __init__(self, weights: dict = None) -> None:
        """
        Initialise the model with optional custom weights.

        Parameters
        ----------
        weights : dict, optional
            Keys must include "rainfall", "terrain", "historical", "drainage".
            Values must sum to 1.0 (±0.001 tolerance).

        Raises
        ------
        ValueError
            If weights do not sum to 1.0 within tolerance.
        """
        self.weights: Dict[str, float] = dict(weights) if weights is not None else dict(self._DEFAULT_WEIGHTS)
        self._validate_weights()

    def _validate_weights(self) -> None:
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"FloodRiskModel weights must sum to 1.0 (got {total:.4f}). "
                f"Weights: {self.weights}"
            )

    def risk_label(self, score: float) -> str:
        """
        Converts a numeric risk score to a categorical label.

        Thresholds (consistent with IMD severity categories):
            < 0.30  -> "Low"
            < 0.60  -> "Moderate"
            < 0.80  -> "High"
            >= 0.80 -> "Critical"
        """
        if score < 0.30:
            return "Low"
        elif score < 0.60:
            return "Moderate"
        elif score < 0.80:
            return "High"
        else:
            return "Critical"

    def predict(self, feature_vector) -> RiskAssessment:
        """
        Produce a RiskAssessment from a FloodFeatureVector.

        Parameters
        ----------
        feature_vector : FloodFeatureVector
            Assembled feature vector for the target location.

        Returns
        -------
        RiskAssessment
        """
        r_rainfall = feature_vector.rainfall.compute_risk_score()
        r_terrain = feature_vector.terrain.compute_risk_score()
        r_historical = feature_vector.historical.compute_risk_score()
        r_drainage = 0.0   # placeholder; no drainage sub-model connected yet

        composite = (
            self.weights["rainfall"] * r_rainfall
            + self.weights["terrain"] * r_terrain
            + self.weights["historical"] * r_historical
            + self.weights["drainage"] * r_drainage
        )
        composite = float(np.clip(composite, 0.0, 1.0))

        return RiskAssessment(
            scenario_id=f"{feature_vector.lat:.4f}_{feature_vector.lon:.4f}_{feature_vector.timestamp.isoformat()}",
            lat=feature_vector.lat,
            lon=feature_vector.lon,
            timestamp=feature_vector.timestamp,
            composite_risk_score=composite,
            risk_label=self.risk_label(composite),
            rainfall_risk=r_rainfall,
            terrain_risk=r_terrain,
            historical_risk=r_historical,
            drainage_risk=r_drainage,
            weights_used=dict(self.weights),
            data_gaps=feature_vector.missing_inputs(),
            model_version=self.MODEL_VERSION,
        )

    def predict_batch(self, feature_vectors: list) -> List[RiskAssessment]:
        """
        Apply predict() to a list of FloodFeatureVectors.

        Parameters
        ----------
        feature_vectors : list of FloodFeatureVector

        Returns
        -------
        list of RiskAssessment
        """
        return [self.predict(fv) for fv in feature_vectors]

    def describe_risk(self, assessment: RiskAssessment) -> str:
        """
        Returns a human-readable summary of the risk assessment.

        Example output:
            "Moderate flood risk (score: 0.52). Driven primarily by heavy
             antecedent rainfall over 72 hours. Terrain shows low-lying area
             with moderate flow accumulation. Historical data: 2 known flood
             events. Missing data: drainage."
        """
        score = assessment.composite_risk_score
        label = assessment.risk_label

        # Identify dominant driver
        component_scores = {
            "rainfall": assessment.rainfall_risk * self.weights["rainfall"],
            "terrain": assessment.terrain_risk * self.weights["terrain"],
            "historical": assessment.historical_risk * self.weights["historical"],
            "drainage": assessment.drainage_risk * self.weights["drainage"],
        }
        dominant = max(component_scores, key=component_scores.get)

        driver_descriptions = {
            "rainfall": "heavy antecedent rainfall over 72 hours",
            "terrain": "unfavourable terrain (low elevation / high flow accumulation)",
            "historical": "high historical flood susceptibility for this area",
            "drainage": "poor drainage infrastructure",
        }

        terrain_desc = ""
        if assessment.terrain_risk > 0.5:
            terrain_desc = "Terrain shows low-lying area with high flow accumulation."
        elif assessment.terrain_risk > 0.25:
            terrain_desc = "Terrain shows low-lying area with moderate flow accumulation."
        else:
            terrain_desc = "Terrain is relatively elevated with good drainage gradients."

        missing_str = ", ".join(assessment.data_gaps) if assessment.data_gaps else "none"

        return (
            f"{label} flood risk (score: {score:.2f}). "
            f"Driven primarily by {driver_descriptions.get(dominant, dominant)}. "
            f"{terrain_desc} "
            f"Missing data: {missing_str}."
        )


def quick_risk_estimate(
    lat: float,
    lon: float,
    rainfall_mm_hr: float,
    elevation_m: float = None,
) -> float:
    """
    Fast single-point flood risk estimate using only rainfall and elevation.

    This is a lightweight screening function for situations where full
    feature vectors are not available. It does NOT use terrain flow
    accumulation, depression mapping, historical records, or drainage data.

    Parameters
    ----------
    lat, lon        : Geographic coordinates (informational only).
    rainfall_mm_hr  : Current rainfall rate in mm/hr.
    elevation_m     : Elevation in metres. If None, elevation component is skipped.

    Returns
    -------
    float
        Rough risk score in [0, 1].
    """
    # Rainfall component: normalise by IMD extremely heavy threshold
    rainfall_component = float(np.clip(rainfall_mm_hr / 115.6, 0.0, 1.0))

    if elevation_m is not None:
        # Simple elevation risk: assume 50 m as regional max for low-lying areas
        elevation_component = 1.0 - float(np.clip(elevation_m / 50.0, 0.0, 1.0))
        score = 0.6 * rainfall_component + 0.4 * elevation_component
    else:
        score = rainfall_component

    return float(np.clip(score, 0.0, 1.0))
