"""Metrics for comparing flood-aware routes with baseline routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RouteMetrics:
    """Comparison metrics for a baseline and an alternative route."""

    distance_change_pct: float
    travel_time_change_pct: float
    hazard_reduction_pct: float
    high_hazard_segments_change: int


def _percentage_change(value: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0 if value == 0 else float("inf")
    return (value - baseline) / baseline * 100.0


def _hazard_reduction(baseline: float, alternative: float) -> float:
    if baseline == 0:
        return 0.0 if alternative == 0 else float("-inf")
    return (baseline - alternative) / baseline * 100.0


def compare_routes(baseline, alternative) -> RouteMetrics:
    """Compare two route results using their public numeric attributes.

    Positive distance/time changes indicate a detour. Positive hazard reduction
    indicates that the alternative exposes responders to less hazard.
    """
    if baseline is None or alternative is None:
        raise ValueError("Both baseline and alternative routes are required")

    return RouteMetrics(
        distance_change_pct=_percentage_change(
            alternative.total_distance_m, baseline.total_distance_m
        ),
        travel_time_change_pct=_percentage_change(
            alternative.total_travel_time_s, baseline.total_travel_time_s
        ),
        hazard_reduction_pct=_hazard_reduction(
            baseline.total_hazard_exposure,
            alternative.total_hazard_exposure,
        ),
        high_hazard_segments_change=(
            alternative.high_hazard_segments - baseline.high_hazard_segments
        ),
    )
