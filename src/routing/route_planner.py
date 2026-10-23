"""
Flood-aware route planner using Dijkstra's algorithm / NetworkX shortest path.

Computes and compares:
1. Baseline: shortest distance route (ignores flood risk)
2. Fastest: shortest travel time route (ignores flood risk)
3. Flood-aware: minimizes C_e = T_e + λH_e + μU_e while removing blocked edges

For each route, reports:
- Distance (km)
- Estimated travel time (minutes)
- Number of high-hazard segments traversed
- Total hazard exposure score
- Whether route is feasible (may be None if no safe path exists)

The routing algorithm does NOT guarantee physical safety.
Recommendations should be verified by qualified emergency responders.
"""
import networkx as nx
import osmnx as ox
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RouteResult:
    """Full details for a single computed route."""

    route_type: str                      # "baseline_distance" | "baseline_time" | "flood_aware"
    start_node: int
    end_node: int
    start_coords: Tuple[float, float]    # (lat, lon)
    end_coords: Tuple[float, float]      # (lat, lon)
    node_path: List[int]
    edge_path: List[Tuple]
    total_distance_m: float
    total_travel_time_s: float
    high_hazard_segments: int
    total_hazard_exposure: float         # sum of flood_risk along path
    max_segment_risk: float
    is_feasible: bool
    infeasible_reason: str = ""
    geometry: Optional[object] = None   # Shapely LineString
    route_warnings: List[str] = field(default_factory=list)
    computed_at: datetime = field(default_factory=datetime.utcnow)

    def summary_dict(self) -> dict:
        """
        Return a flat dict suitable for display in a DataFrame row.

        Returns
        -------
        dict
        """
        return {
            "route_type": self.route_type,
            "start_node": self.start_node,
            "end_node": self.end_node,
            "start_lat": self.start_coords[0],
            "start_lon": self.start_coords[1],
            "end_lat": self.end_coords[0],
            "end_lon": self.end_coords[1],
            "distance_km": round(self.total_distance_m / 1000.0, 3),
            "travel_time_min": round(self.total_travel_time_s / 60.0, 2),
            "high_hazard_segments": self.high_hazard_segments,
            "total_hazard_exposure": round(self.total_hazard_exposure, 4),
            "max_segment_risk": round(self.max_segment_risk, 4),
            "is_feasible": self.is_feasible,
            "infeasible_reason": self.infeasible_reason,
            "num_nodes_in_path": len(self.node_path),
            "warnings": "; ".join(self.route_warnings),
            "computed_at": self.computed_at.isoformat(),
        }


@dataclass
class RouteComparison:
    """Aggregated comparison of all three route variants."""

    baseline_distance: Optional[RouteResult]
    baseline_time: Optional[RouteResult]
    flood_aware: Optional[RouteResult]
    start_coords: Tuple[float, float]
    end_coords: Tuple[float, float]
    recommendation: str                  # textual recommendation for responders
    detour_distance_pct: float           # how much longer the flood-aware route is
    hazard_reduction_pct: float          # how much hazard exposure is reduced

    def comparison_dataframe(self) -> pd.DataFrame:
        """
        Build a summary DataFrame with one row per route type.

        Columns: type, distance_km, travel_time_min, high_hazard_segments,
        total_hazard, feasible.

        Returns
        -------
        pd.DataFrame
        """
        rows = []
        for route in [self.baseline_distance, self.baseline_time, self.flood_aware]:
            if route is None:
                continue
            rows.append(
                {
                    "type": route.route_type,
                    "distance_km": round(route.total_distance_m / 1000.0, 3),
                    "travel_time_min": round(route.total_travel_time_s / 60.0, 2),
                    "high_hazard_segments": route.high_hazard_segments,
                    "total_hazard": round(route.total_hazard_exposure, 4),
                    "feasible": route.is_feasible,
                }
            )
        return pd.DataFrame(rows)


class RoutePlanner:
    """
    Plans and compares flood-aware rescue routes.

    Three route variants are computed for every origin–destination pair:

    * **baseline_distance** – classic shortest-path by road distance (``length``
      attribute).  Ignores flood risk entirely.
    * **baseline_time** – shortest-path by free-flow travel time (``travel_time``
      attribute).  Ignores flood risk entirely.
    * **flood_aware** – shortest-path using the composite hazard cost
      ``hazard_cost`` (requires blocked edges to have been removed from the
      feasible graph beforehand).

    The planner operates on a :class:`~routing.graph_builder.GraphBuildResult`
    so that it has access to both the full annotated graph (for baseline routes)
    and the feasible graph (for the flood-aware route).
    """

    def __init__(self, cost_fn=None) -> None:
        """
        Parameters
        ----------
        cost_fn : HazardCostFunction, optional
            Used to determine hazard thresholds when classifying segments.
            If *None*, a default instance is created internally.
        """
        if cost_fn is None:
            from .hazard_cost import HazardCostFunction
            self.cost_fn = HazardCostFunction()
        else:
            self.cost_fn = cost_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan_routes(
        self,
        graph_result,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
    ) -> RouteComparison:
        """
        Plan all three route variants for an origin–destination pair.

        Parameters
        ----------
        graph_result : GraphBuildResult
            The three-layer graph produced by :class:`FloodAwareGraphBuilder`.
        start_lat, start_lon : float
            Geographic coordinates of the origin.
        end_lat, end_lon : float
            Geographic coordinates of the destination.

        Returns
        -------
        RouteComparison
        """
        G_hazard: nx.MultiDiGraph = graph_result.graph_hazard
        G_feasible: nx.MultiDiGraph = graph_result.graph_feasible

        # Snap coordinates to nearest nodes in the annotated graph.
        start_node = ox.distance.nearest_nodes(G_hazard, X=start_lon, Y=start_lat)
        end_node = ox.distance.nearest_nodes(G_hazard, X=end_lon, Y=end_lat)

        start_coords = (start_lat, start_lon)
        end_coords = (end_lat, end_lon)

        # Plan routes.
        baseline_dist = self._plan_single_route(
            G_hazard, start_node, end_node, weight="length",
            route_type="baseline_distance",
            start_coords=start_coords, end_coords=end_coords,
        )
        baseline_time = self._plan_single_route(
            G_hazard, start_node, end_node, weight="travel_time",
            route_type="baseline_time",
            start_coords=start_coords, end_coords=end_coords,
        )
        flood_aware = self._plan_single_route(
            G_feasible, start_node, end_node, weight="hazard_cost",
            route_type="flood_aware",
            start_coords=start_coords, end_coords=end_coords,
        )

        # Derive comparison statistics.
        detour_pct, hazard_reduction_pct = self._compute_comparison_stats(
            baseline_dist, flood_aware
        )

        comparison = RouteComparison(
            baseline_distance=baseline_dist,
            baseline_time=baseline_time,
            flood_aware=flood_aware,
            start_coords=start_coords,
            end_coords=end_coords,
            recommendation="",          # filled in below
            detour_distance_pct=detour_pct,
            hazard_reduction_pct=hazard_reduction_pct,
        )
        comparison.recommendation = self._generate_recommendation(comparison)

        return comparison

    def find_nearest_safe_shelter(
        self,
        G: nx.MultiDiGraph,
        start_lat: float,
        start_lon: float,
        shelters: List[dict],
        max_shelters_to_try: int = 5,
    ) -> Optional[RouteResult]:
        """
        Route from a start location to the nearest reachable safe shelter.

        Parameters
        ----------
        G : nx.MultiDiGraph
            The **feasible** graph (blocked edges already removed).
        start_lat, start_lon : float
            Geographic coordinates of the origin.
        shelters : List[dict]
            Each shelter dict must contain at least ``lat``, ``lon``, and
            ``is_safe`` (bool) keys.
        max_shelters_to_try : int
            Maximum number of safe shelters to attempt before giving up.
            Shelters are sorted by straight-line distance from the origin
            so that the closest candidates are tried first.

        Returns
        -------
        Optional[RouteResult]
            The lowest-cost (by ``hazard_cost``) route to a reachable safe
            shelter, or *None* if no safe shelter can be reached.
        """
        start_coords = (start_lat, start_lon)
        start_node = ox.distance.nearest_nodes(G, X=start_lon, Y=start_lat)

        # Filter to safe shelters only.
        safe_shelters = [s for s in shelters if s.get("is_safe", False)]
        if not safe_shelters:
            return None

        # Sort by straight-line (Euclidean) distance to start — degrees are
        # fine for ranking purposes.
        safe_shelters = sorted(
            safe_shelters,
            key=lambda s: np.hypot(s["lat"] - start_lat, s["lon"] - start_lon),
        )

        best_route: Optional[RouteResult] = None

        for shelter in safe_shelters[:max_shelters_to_try]:
            shelter_lat = float(shelter["lat"])
            shelter_lon = float(shelter["lon"])
            end_node = ox.distance.nearest_nodes(G, X=shelter_lon, Y=shelter_lat)

            if end_node == start_node:
                # Already at (or snapped to the same node as) the shelter.
                continue

            route = self._plan_single_route(
                G,
                source=start_node,
                target=end_node,
                weight="hazard_cost",
                route_type="flood_aware",
                start_coords=start_coords,
                end_coords=(shelter_lat, shelter_lon),
            )

            if route is None or not route.is_feasible:
                continue

            if best_route is None or route.total_hazard_exposure < best_route.total_hazard_exposure:
                best_route = route

        return best_route

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _plan_single_route(
        self,
        G: nx.MultiDiGraph,
        source: int,
        target: int,
        weight: str,
        route_type: str,
        start_coords: Tuple[float, float] = (0.0, 0.0),
        end_coords: Tuple[float, float] = (0.0, 0.0),
    ) -> Optional[RouteResult]:
        """
        Compute a single shortest-path route.

        Parameters
        ----------
        G : nx.MultiDiGraph
        source, target : int
            NetworkX node identifiers.
        weight : str
            Edge attribute to use as the path cost.
        route_type : str
            Label stored in the returned :class:`RouteResult`.
        start_coords, end_coords : Tuple[float, float]
            Geographic coordinates stored for display purposes.

        Returns
        -------
        Optional[RouteResult]
            *None* is returned only on unexpected internal errors; infeasible
            routes are returned as :class:`RouteResult` objects with
            ``is_feasible=False``.
        """
        try:
            node_path: List[int] = nx.shortest_path(
                G, source=source, target=target, weight=weight
            )
        except nx.NetworkXNoPath:
            return RouteResult(
                route_type=route_type,
                start_node=source,
                end_node=target,
                start_coords=start_coords,
                end_coords=end_coords,
                node_path=[],
                edge_path=[],
                total_distance_m=0.0,
                total_travel_time_s=0.0,
                high_hazard_segments=0,
                total_hazard_exposure=0.0,
                max_segment_risk=0.0,
                is_feasible=False,
                infeasible_reason=(
                    f"No path found from node {source} to {target} "
                    f"using weight='{weight}'. The destination may be "
                    "unreachable due to blocked edges or graph disconnection."
                ),
            )
        except nx.NodeNotFound as exc:
            return RouteResult(
                route_type=route_type,
                start_node=source,
                end_node=target,
                start_coords=start_coords,
                end_coords=end_coords,
                node_path=[],
                edge_path=[],
                total_distance_m=0.0,
                total_travel_time_s=0.0,
                high_hazard_segments=0,
                total_hazard_exposure=0.0,
                max_segment_risk=0.0,
                is_feasible=False,
                infeasible_reason=f"Node not found in graph: {exc}",
            )

        metrics = self._compute_route_metrics(G, node_path)
        edge_path = self._extract_edge_path(G, node_path)
        geometry = self._extract_route_geometry(G, node_path)

        warnings: List[str] = []
        if metrics["high_hazard_segments"] > 0:
            warnings.append(
                f"Route passes through {metrics['high_hazard_segments']} "
                "high-hazard segment(s). Verify with field teams before dispatch."
            )
        if metrics["max_segment_risk"] >= self.cost_fn.high_hazard_threshold:
            warnings.append(
                f"Maximum segment risk is {metrics['max_segment_risk']:.2f} — "
                "consider alternative routes."
            )

        return RouteResult(
            route_type=route_type,
            start_node=source,
            end_node=target,
            start_coords=start_coords,
            end_coords=end_coords,
            node_path=node_path,
            edge_path=edge_path,
            total_distance_m=metrics["distance_m"],
            total_travel_time_s=metrics["travel_time_s"],
            high_hazard_segments=metrics["high_hazard_segments"],
            total_hazard_exposure=metrics["total_hazard_exposure"],
            max_segment_risk=metrics["max_segment_risk"],
            is_feasible=True,
            geometry=geometry,
            route_warnings=warnings,
        )

    def _compute_route_metrics(
        self, G: nx.MultiDiGraph, node_path: List[int]
    ) -> Dict:
        """
        Walk the node path and accumulate route statistics.

        Parameters
        ----------
        G : nx.MultiDiGraph
        node_path : List[int]
            Ordered list of node IDs representing the route.

        Returns
        -------
        dict
            Keys: ``distance_m``, ``travel_time_s``, ``high_hazard_segments``,
            ``total_hazard_exposure``, ``max_segment_risk``.
        """
        distance_m = 0.0
        travel_time_s = 0.0
        high_hazard_segments = 0
        total_hazard_exposure = 0.0
        max_segment_risk = 0.0

        for i in range(len(node_path) - 1):
            u = node_path[i]
            v = node_path[i + 1]

            # Pick the edge with the lowest weight (handles parallel edges).
            edge_data = self._best_edge_data(G, u, v)

            distance_m += float(edge_data.get("length", 0.0))
            travel_time_s += float(edge_data.get("travel_time", 30.0))

            flood_risk = float(edge_data.get("flood_risk", 0.0))
            total_hazard_exposure += flood_risk

            if flood_risk > max_segment_risk:
                max_segment_risk = flood_risk

            if flood_risk >= self.cost_fn.high_hazard_threshold:
                high_hazard_segments += 1

        return {
            "distance_m": distance_m,
            "travel_time_s": travel_time_s,
            "high_hazard_segments": high_hazard_segments,
            "total_hazard_exposure": total_hazard_exposure,
            "max_segment_risk": max_segment_risk,
        }

    def _extract_edge_path(
        self, G: nx.MultiDiGraph, node_path: List[int]
    ) -> List[Tuple]:
        """
        Convert a node path to an ordered list of ``(u, v, key)`` tuples.

        Parameters
        ----------
        G : nx.MultiDiGraph
        node_path : List[int]

        Returns
        -------
        List[Tuple]
        """
        edge_path = []
        for i in range(len(node_path) - 1):
            u = node_path[i]
            v = node_path[i + 1]
            # Select the parallel edge with the lowest hazard_cost (or length).
            best_key = min(
                G[u][v],
                key=lambda k: G[u][v][k].get("hazard_cost", G[u][v][k].get("length", 0.0)),
            )
            edge_path.append((u, v, best_key))
        return edge_path

    def _extract_route_geometry(
        self, G: nx.MultiDiGraph, node_path: List[int]
    ) -> Optional[object]:
        """
        Build a Shapely LineString tracing the full route.

        Parameters
        ----------
        G : nx.MultiDiGraph
        node_path : List[int]

        Returns
        -------
        shapely.geometry.LineString or None
            *None* if the path is empty or geometry cannot be constructed.
        """
        if len(node_path) < 2:
            return None

        coords: List[Tuple[float, float]] = []

        for i in range(len(node_path) - 1):
            u = node_path[i]
            v = node_path[i + 1]
            best_key = min(
                G[u][v],
                key=lambda k: G[u][v][k].get("hazard_cost", G[u][v][k].get("length", 0.0)),
            )
            edge_data = G[u][v][best_key]
            geometry = edge_data.get("geometry", None)

            if geometry is not None:
                try:
                    edge_coords = list(geometry.coords)
                    if coords and edge_coords and coords[-1] == edge_coords[0]:
                        edge_coords = edge_coords[1:]
                    coords.extend(edge_coords)
                    continue
                except Exception:
                    pass

            # Fallback: straight line between node positions.
            u_data = G.nodes[u]
            v_data = G.nodes[v]
            u_pt = (float(u_data.get("x", 0.0)), float(u_data.get("y", 0.0)))
            v_pt = (float(v_data.get("x", 0.0)), float(v_data.get("y", 0.0)))

            if not coords:
                coords.append(u_pt)
            coords.append(v_pt)

        try:
            return LineString(coords) if len(coords) >= 2 else None
        except Exception:
            return None

    def _generate_recommendation(self, comparison: "RouteComparison") -> str:
        """
        Generate a human-readable route recommendation for emergency responders.

        Parameters
        ----------
        comparison : RouteComparison

        Returns
        -------
        str
        """
        flood_aware = comparison.flood_aware
        baseline_dist = comparison.baseline_distance

        if flood_aware is None or not flood_aware.is_feasible:
            reason = (
                flood_aware.infeasible_reason
                if flood_aware is not None
                else "No flood-aware route computed."
            )
            if baseline_dist is not None and baseline_dist.is_feasible:
                return (
                    "⚠ CAUTION: No flood-safe route is available. "
                    f"Reason: {reason} "
                    "The baseline distance route may pass through hazardous areas. "
                    "Dispatch only with explicit authorization and appropriate vehicles."
                )
            return (
                "✗ CRITICAL: No feasible route found. "
                f"Reason: {reason} "
                "The destination cannot be reached — consider air or water-based rescue."
            )

        lines: List[str] = []

        if baseline_dist is not None and baseline_dist.is_feasible:
            detour_km = (
                flood_aware.total_distance_m - baseline_dist.total_distance_m
            ) / 1000.0
            detour_pct = comparison.detour_distance_pct

            if detour_km <= 0:
                lines.append(
                    "✓ The flood-aware route is the same length as or shorter than "
                    "the baseline route."
                )
            else:
                lines.append(
                    f"✓ The flood-aware route adds {detour_km:.1f} km "
                    f"({detour_pct:.1f}% longer than the baseline distance route)"
                )

            hazard_red = comparison.hazard_reduction_pct
            if hazard_red > 0:
                lines.append(
                    f"  and reduces hazard exposure by {hazard_red:.1f}%."
                )
            else:
                lines.append("  with similar hazard exposure to the baseline route.")

        seg_count = flood_aware.high_hazard_segments
        if seg_count == 0:
            lines.append("  No high-hazard segments are traversed. ✓")
        else:
            lines.append(
                f"  ⚠ The route passes through {seg_count} high-hazard segment(s) — "
                "proceed with caution and confirm conditions with field teams."
            )

        lines.append("RECOMMENDED: Use the flood-aware route.")

        return " ".join(lines)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _best_edge_data(
        self, G: nx.MultiDiGraph, u: int, v: int
    ) -> dict:
        """
        Return the data dict for the lowest-cost parallel edge between u and v.

        Falls back to the first available edge if ``hazard_cost`` is not set.
        """
        edges = G[u][v]
        best_key = min(
            edges,
            key=lambda k: edges[k].get("hazard_cost", edges[k].get("length", 0.0)),
        )
        return edges[best_key]

    @staticmethod
    def _compute_comparison_stats(
        baseline: Optional[RouteResult],
        flood_aware: Optional[RouteResult],
    ) -> Tuple[float, float]:
        """
        Compute detour distance percentage and hazard reduction percentage.

        Returns
        -------
        Tuple[float, float]
            ``(detour_distance_pct, hazard_reduction_pct)``
        """
        detour_pct = 0.0
        hazard_reduction_pct = 0.0

        if (
            baseline is not None
            and baseline.is_feasible
            and flood_aware is not None
            and flood_aware.is_feasible
        ):
            if baseline.total_distance_m > 0:
                detour_pct = (
                    (flood_aware.total_distance_m - baseline.total_distance_m)
                    / baseline.total_distance_m
                    * 100.0
                )

            if baseline.total_hazard_exposure > 0:
                hazard_reduction_pct = (
                    (baseline.total_hazard_exposure - flood_aware.total_hazard_exposure)
                    / baseline.total_hazard_exposure
                    * 100.0
                )

        return round(detour_pct, 2), round(hazard_reduction_pct, 2)
