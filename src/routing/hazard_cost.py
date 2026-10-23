"""
Hazard-aware edge cost function for flood rescue routing.

Cost model: C_e = T_e + λ*H_e + μ*U_e

Where:
    T_e: estimated travel time for edge e (seconds)
    H_e: flood hazard penalty (seconds equivalent) based on flood risk [0-1]
    U_e: uncertainty penalty based on data quality
    λ: relative importance of hazard vs travel time (from config)
    μ: relative importance of uncertainty

Hard constraint: edges with flood_risk > IMPASSABLE_THRESHOLD are removed
from the feasible graph entirely, not merely penalized.

IMPORTANT: The road passability thresholds used here are based on relative
risk scores, not calibrated water depth measurements. Vehicle passability
depends on depth, flow velocity, road type, vehicle clearance, and other
factors. These thresholds are for planning support only.
"""
import numpy as np
import networkx as nx
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class EdgeCost:
    """Full cost breakdown for a single graph edge."""

    edge_id: tuple  # (u, v, key)
    travel_time_s: float
    hazard_penalty_s: float
    uncertainty_penalty_s: float
    total_cost: float
    flood_risk: float
    is_blocked: bool
    cost_breakdown: Dict[str, float]


class HazardCostFunction:
    """
    Computes hazard-aware edge costs for flood rescue routing.

    Cost model:
        C_e = T_e + λ * H_e + μ * U_e

    Edges whose flood_risk >= impassable_threshold receive an infinite cost
    and are treated as hard-blocked — they are removed from the feasible graph.
    """

    def __init__(
        self,
        lambda_hazard: float = 2.0,
        mu_uncertainty: float = 0.5,
        impassable_threshold: float = 0.85,
        high_hazard_threshold: float = 0.65,
    ) -> None:
        """
        Parameters
        ----------
        lambda_hazard : float
            Relative weight given to the flood-hazard penalty vs raw travel time.
            Higher values make the router more avoidance-oriented.
        mu_uncertainty : float
            Relative weight given to the uncertainty penalty.
        impassable_threshold : float
            flood_risk >= this value → edge is treated as completely impassable.
        high_hazard_threshold : float
            flood_risk >= this value (but < impassable_threshold) triggers an
            amplified non-linear hazard penalty.
        """
        if not (0.0 < impassable_threshold <= 1.0):
            raise ValueError("impassable_threshold must be in (0, 1].")
        if not (0.0 <= high_hazard_threshold < impassable_threshold):
            raise ValueError(
                "high_hazard_threshold must be in [0, impassable_threshold)."
            )

        self.lambda_hazard = lambda_hazard
        self.mu_uncertainty = mu_uncertainty
        self.impassable_threshold = impassable_threshold
        self.high_hazard_threshold = high_hazard_threshold

    # ------------------------------------------------------------------
    # Core penalty computations
    # ------------------------------------------------------------------

    def compute_hazard_penalty(self, flood_risk: float, travel_time_s: float) -> float:
        """
        Compute the hazard penalty H_e in seconds equivalent.

        Parameters
        ----------
        flood_risk : float
            Normalised flood risk score for the edge [0, 1].
        travel_time_s : float
            Free-flow travel time for the edge in seconds.

        Returns
        -------
        float
            Hazard penalty in seconds (may be ``float('inf')`` for blocked edges).
        """
        flood_risk = float(np.clip(flood_risk, 0.0, 1.0))

        if flood_risk >= self.impassable_threshold:
            # Hard block — treat the edge as completely impassable.
            return float("inf")

        if flood_risk >= self.high_hazard_threshold:
            # Non-linear amplification for high-hazard (but still technically
            # passable) segments.
            penalty = (
                self.lambda_hazard
                * travel_time_s
                * (flood_risk / self.high_hazard_threshold)
                * 3
            )
        else:
            # Linear regime for low-to-moderate risk.
            penalty = self.lambda_hazard * travel_time_s * flood_risk

        return float(penalty)

    def compute_uncertainty_penalty(
        self, data_quality: float, travel_time_s: float
    ) -> float:
        """
        Compute the uncertainty penalty U_e in seconds equivalent.

        Parameters
        ----------
        data_quality : float
            Quality / confidence of the flood-risk estimate for this edge.
            1.0 = fully known / high confidence; 0.0 = entirely unknown.
        travel_time_s : float
            Free-flow travel time for the edge in seconds.

        Returns
        -------
        float
            Uncertainty penalty in seconds.
        """
        data_quality = float(np.clip(data_quality, 0.0, 1.0))
        uncertainty = self.mu_uncertainty * travel_time_s * (1.0 - data_quality)
        return float(uncertainty)

    def compute_edge_cost(
        self,
        travel_time_s: float,
        flood_risk: float,
        data_quality: float = 1.0,
        edge_id: tuple = (0, 0, 0),
    ) -> EdgeCost:
        """
        Assemble the full EdgeCost object for a single edge.

        Parameters
        ----------
        travel_time_s : float
            Free-flow travel time in seconds.
        flood_risk : float
            Normalised flood risk score [0, 1].
        data_quality : float
            Confidence of the flood estimate [0, 1].  Defaults to 1.0.
        edge_id : tuple
            (u, v, key) identifier for the edge.  Defaults to (0, 0, 0).

        Returns
        -------
        EdgeCost
        """
        travel_time_s = max(0.0, float(travel_time_s))
        flood_risk = float(np.clip(flood_risk, 0.0, 1.0))
        data_quality = float(np.clip(data_quality, 0.0, 1.0))

        is_blocked = flood_risk >= self.impassable_threshold

        hazard_penalty_s = self.compute_hazard_penalty(flood_risk, travel_time_s)
        uncertainty_penalty_s = self.compute_uncertainty_penalty(
            data_quality, travel_time_s
        )

        if is_blocked:
            total_cost = float("inf")
        else:
            total_cost = travel_time_s + hazard_penalty_s + uncertainty_penalty_s

        cost_breakdown: Dict[str, float] = {
            "travel_time_s": travel_time_s,
            "hazard_penalty_s": hazard_penalty_s if not is_blocked else float("inf"),
            "uncertainty_penalty_s": uncertainty_penalty_s,
            "lambda_hazard": self.lambda_hazard,
            "mu_uncertainty": self.mu_uncertainty,
            "flood_risk": flood_risk,
            "data_quality": data_quality,
            "total_cost": total_cost,
        }

        return EdgeCost(
            edge_id=edge_id,
            travel_time_s=travel_time_s,
            hazard_penalty_s=hazard_penalty_s if not is_blocked else float("inf"),
            uncertainty_penalty_s=uncertainty_penalty_s,
            total_cost=total_cost,
            flood_risk=flood_risk,
            is_blocked=is_blocked,
            cost_breakdown=cost_breakdown,
        )

    # ------------------------------------------------------------------
    # Graph-level operations
    # ------------------------------------------------------------------

    def apply_to_graph(
        self, G: nx.MultiDiGraph, remove_blocked: bool = True
    ) -> nx.MultiDiGraph:
        """
        Apply the hazard cost function to every edge in the graph.

        For each edge the following attributes are written:
        - ``hazard_cost``        – total cost C_e (float or inf)
        - ``hazard_penalty_s``   – flood hazard component
        - ``uncertainty_penalty_s`` – uncertainty component
        - ``is_blocked``         – True if flood_risk >= impassable_threshold

        Parameters
        ----------
        G : nx.MultiDiGraph
            Input road graph.  Must have ``travel_time`` (seconds) and
            ``flood_risk`` edge attributes (``flood_risk`` defaults to 0.0
            if missing; ``travel_time`` defaults to 30 s if missing).
        remove_blocked : bool
            If *True* (default) all impassable edges are deleted from the
            returned graph.  The original graph is **not** modified.

        Returns
        -------
        nx.MultiDiGraph
            A deep copy of *G* with hazard cost attributes applied and,
            optionally, blocked edges removed.
        """
        G_copy: nx.MultiDiGraph = G.copy()

        edges_to_remove = []

        for u, v, key, data in G_copy.edges(keys=True, data=True):
            travel_time_s: float = float(data.get("travel_time", 30.0))
            flood_risk: float = float(data.get("flood_risk", 0.0))
            data_quality: float = float(data.get("data_quality", 1.0))

            ec = self.compute_edge_cost(
                travel_time_s=travel_time_s,
                flood_risk=flood_risk,
                data_quality=data_quality,
                edge_id=(u, v, key),
            )

            # Write computed attributes back to the copied graph.
            G_copy[u][v][key]["hazard_cost"] = ec.total_cost
            G_copy[u][v][key]["hazard_penalty_s"] = ec.hazard_penalty_s
            G_copy[u][v][key]["uncertainty_penalty_s"] = ec.uncertainty_penalty_s
            G_copy[u][v][key]["is_blocked"] = ec.is_blocked

            if ec.is_blocked and remove_blocked:
                edges_to_remove.append((u, v, key))

        for u, v, key in edges_to_remove:
            G_copy.remove_edge(u, v, key)

        return G_copy

    # ------------------------------------------------------------------
    # Human-readable explanation
    # ------------------------------------------------------------------

    def get_cost_explanation(self, edge_cost: EdgeCost) -> str:
        """
        Return a human-readable explanation of why an edge was penalized.

        Parameters
        ----------
        edge_cost : EdgeCost
            The cost object returned by :meth:`compute_edge_cost`.

        Returns
        -------
        str
        """
        u, v, key = edge_cost.edge_id
        risk_pct = edge_cost.flood_risk * 100.0

        if edge_cost.is_blocked:
            return (
                f"Edge ({u} → {v}, key={key}) is IMPASSABLE. "
                f"Flood risk {risk_pct:.1f}% ≥ impassable threshold "
                f"{self.impassable_threshold * 100:.1f}%. "
                "This edge has been removed from the feasible graph."
            )

        lines = [
            f"Edge ({u} → {v}, key={key}) cost breakdown:",
            f"  Base travel time  : {edge_cost.travel_time_s:.1f} s",
        ]

        if edge_cost.flood_risk >= self.high_hazard_threshold:
            lines.append(
                f"  Hazard penalty    : {edge_cost.hazard_penalty_s:.1f} s "
                f"(HIGH HAZARD — risk {risk_pct:.1f}% ≥ high-hazard threshold "
                f"{self.high_hazard_threshold * 100:.1f}%, amplified ×3)"
            )
        elif edge_cost.flood_risk > 0.0:
            lines.append(
                f"  Hazard penalty    : {edge_cost.hazard_penalty_s:.1f} s "
                f"(risk {risk_pct:.1f}%, λ={self.lambda_hazard})"
            )
        else:
            lines.append("  Hazard penalty    : 0.0 s (no flood risk)")

        dq = edge_cost.cost_breakdown.get("data_quality", 1.0)
        if dq < 1.0:
            lines.append(
                f"  Uncertainty penalty: {edge_cost.uncertainty_penalty_s:.1f} s "
                f"(data quality {dq * 100:.1f}%, μ={self.mu_uncertainty})"
            )
        else:
            lines.append("  Uncertainty penalty: 0.0 s (full data quality)")

        lines.append(f"  ─────────────────────────────────────")
        lines.append(f"  Total cost        : {edge_cost.total_cost:.1f} s")

        return "\n".join(lines)
