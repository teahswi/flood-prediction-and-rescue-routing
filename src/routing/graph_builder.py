"""
Builds flood-aware road graphs for routing.
Combines OSMnx road network with flood risk predictions to create
a hazard-weighted graph suitable for rescue routing.
"""
import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Tuple, Optional, List, Dict, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .hazard_cost import HazardCostFunction


@dataclass
class GraphBuildResult:
    """
    Container returned by :class:`FloodAwareGraphBuilder` after building
    the road graph layers.
    """

    graph_original: nx.MultiDiGraph      # unmodified road graph
    graph_hazard: nx.MultiDiGraph        # flood-risk annotated graph (edges present)
    graph_feasible: nx.MultiDiGraph      # blocked edges removed
    num_blocked_edges: int
    num_high_hazard_edges: int
    connectivity_info: dict
    build_timestamp: str


class FloodAwareGraphBuilder:
    """
    Constructs a three-layer road-graph product:

    1. ``graph_original``  — the raw OSMnx road graph, untouched.
    2. ``graph_hazard``    — same graph with ``flood_risk``, ``hazard_cost``,
       ``is_blocked``, and related attributes added to every edge.
    3. ``graph_feasible``  — ``graph_hazard`` with impassable edges removed,
       ready for Dijkstra routing.
    """

    def __init__(self, cost_fn: Optional[HazardCostFunction] = None) -> None:
        """
        Parameters
        ----------
        cost_fn : HazardCostFunction, optional
            Custom cost function instance.  A default-parameterised instance
            is created when *None* is passed.
        """
        self.cost_fn: HazardCostFunction = (
            cost_fn if cost_fn is not None else HazardCostFunction()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        road_graph: nx.MultiDiGraph,
        flood_risk_fn: Callable[[float, float], float],
        simulation_result=None,
    ) -> GraphBuildResult:
        """
        Build all three graph layers.

        Parameters
        ----------
        road_graph : nx.MultiDiGraph
            OSMnx road graph (must have node ``x``/``y`` attributes in WGS-84).
        flood_risk_fn : Callable[[float, float], float]
            Function ``f(lat, lon) -> float`` that returns a flood-risk score
            in *[0, 1]* for an arbitrary geographic coordinate.  Used when
            *simulation_result* is *None*.
        simulation_result : optional
            Structured simulation output (e.g. from the flood model).  When
            supplied, ``_flood_risk_from_simulation`` is used for interpolation
            instead of calling *flood_risk_fn* directly.

        Returns
        -------
        GraphBuildResult
        """
        # Step 1 – keep the pristine original.
        graph_original: nx.MultiDiGraph = road_graph.copy()

        # Step 2 & 3 – annotate a working copy with flood-risk scores.
        graph_hazard: nx.MultiDiGraph = road_graph.copy()

        num_blocked = 0
        num_high_hazard = 0

        for u, v, key, data in graph_hazard.edges(keys=True, data=True):
            lat, lon = self._get_edge_midpoint(graph_hazard, u, v, key)

            # Step 3 – query flood risk.
            if simulation_result is not None:
                flood_risk = self._flood_risk_from_simulation(
                    simulation_result, lat, lon
                )
            else:
                try:
                    flood_risk = float(flood_risk_fn(lat, lon))
                except Exception:
                    flood_risk = 0.0

            flood_risk = float(np.clip(flood_risk, 0.0, 1.0))

            is_blocked = flood_risk >= self.cost_fn.impassable_threshold
            is_high_hazard = (
                not is_blocked
                and flood_risk >= self.cost_fn.high_hazard_threshold
            )

            # Step 4 – write risk attributes.
            data["flood_risk"] = flood_risk
            data["is_blocked"] = is_blocked

            travel_time_s = float(data.get("travel_time", 30.0))
            data["hazard_penalty"] = self.cost_fn.compute_hazard_penalty(
                flood_risk, travel_time_s
            )

            if is_blocked:
                num_blocked += 1
            if is_high_hazard:
                num_high_hazard += 1

        # Step 5 – apply full cost function (sets ``hazard_cost`` attribute).
        # We pass remove_blocked=False here so graph_hazard retains all edges.
        graph_hazard = self.cost_fn.apply_to_graph(graph_hazard, remove_blocked=False)

        # Step 6 – build feasible graph (blocked edges removed).
        graph_feasible = self.cost_fn.apply_to_graph(
            graph_hazard, remove_blocked=True
        )

        # Step 7 – connectivity diagnostics.
        connectivity_info = self._check_connectivity(graph_feasible)

        build_timestamp = datetime.now(tz=timezone.utc).isoformat()

        return GraphBuildResult(
            graph_original=graph_original,
            graph_hazard=graph_hazard,
            graph_feasible=graph_feasible,
            num_blocked_edges=num_blocked,
            num_high_hazard_edges=num_high_hazard,
            connectivity_info=connectivity_info,
            build_timestamp=build_timestamp,
        )

    def update_graph_with_new_risk(
        self,
        build_result: GraphBuildResult,
        flood_risk_fn: Callable[[float, float], float],
        simulation_result=None,
    ) -> GraphBuildResult:
        """
        Re-run risk assignment on an existing build result.

        Useful for real-time updates when flood predictions change without
        needing to re-download the road network from OSMnx.

        Parameters
        ----------
        build_result : GraphBuildResult
            Previously built result whose ``graph_original`` is used as the
            source.
        flood_risk_fn : Callable[[float, float], float]
            Updated flood-risk query function.
        simulation_result : optional
            Updated simulation output, if available.

        Returns
        -------
        GraphBuildResult
            A freshly built result based on the original road graph.
        """
        return self.build(
            road_graph=build_result.graph_original,
            flood_risk_fn=flood_risk_fn,
            simulation_result=simulation_result,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_edge_midpoint(
        self,
        G: nx.MultiDiGraph,
        u: int,
        v: int,
        key: int,
    ) -> Tuple[float, float]:
        """
        Return the geographic midpoint of an edge as ``(lat, lon)``.

        If the edge has a Shapely *geometry* attribute the true midpoint of
        the linestring is used; otherwise the average of the two endpoint
        node coordinates is returned.

        Parameters
        ----------
        G : nx.MultiDiGraph
        u, v, key : node / edge identifiers

        Returns
        -------
        Tuple[float, float]
            ``(latitude, longitude)`` in WGS-84 degrees.
        """
        edge_data = G[u][v][key]
        geometry = edge_data.get("geometry", None)

        if geometry is not None:
            # Use true midpoint of the linestring.
            try:
                mid = geometry.interpolate(0.5, normalized=True)
                # OSMnx stores x=lon, y=lat in the projected / geographic CRS.
                return float(mid.y), float(mid.x)
            except Exception:
                pass  # Fall through to node-average fallback.

        # Fallback: average of endpoint node coordinates.
        u_data = G.nodes[u]
        v_data = G.nodes[v]

        u_lat = float(u_data.get("y", 0.0))
        u_lon = float(u_data.get("x", 0.0))
        v_lat = float(v_data.get("y", 0.0))
        v_lon = float(v_data.get("x", 0.0))

        return (u_lat + v_lat) / 2.0, (u_lon + v_lon) / 2.0

    def _flood_risk_from_simulation(
        self,
        simulation_result,
        lat: float,
        lon: float,
    ) -> float:
        """
        Interpolate flood risk from a simulation-result grid at ``(lat, lon)``.

        The method attempts scipy bilinear interpolation first; if that fails
        or the necessary attributes are absent it falls back to a nearest-
        neighbour lookup, and finally returns 0.0 on any remaining error.

        Parameters
        ----------
        simulation_result : object
            Expected to expose either:
            * ``lats``, ``lons``, ``flood_risk`` (2-D numpy arrays), or
            * ``interpolate(lat, lon) -> float`` callable.
        lat : float
        lon : float

        Returns
        -------
        float
            Flood risk score in [0, 1].
        """
        # If the simulation object provides a direct callable interface, use it.
        if callable(getattr(simulation_result, "interpolate", None)):
            try:
                return float(np.clip(simulation_result.interpolate(lat, lon), 0.0, 1.0))
            except Exception:
                pass

        # Try grid-based scipy interpolation.
        try:
            from scipy.interpolate import RegularGridInterpolator

            sim_lats: np.ndarray = np.asarray(simulation_result.lats, dtype=float)
            sim_lons: np.ndarray = np.asarray(simulation_result.lons, dtype=float)
            risk_grid: np.ndarray = np.asarray(
                simulation_result.flood_risk, dtype=float
            )

            # RegularGridInterpolator requires strictly ascending axes.
            if sim_lats[0] > sim_lats[-1]:
                sim_lats = sim_lats[::-1]
                risk_grid = risk_grid[::-1, :]
            if sim_lons[0] > sim_lons[-1]:
                sim_lons = sim_lons[::-1]
                risk_grid = risk_grid[:, ::-1]

            interp = RegularGridInterpolator(
                (sim_lats, sim_lons),
                risk_grid,
                method="linear",
                bounds_error=False,
                fill_value=0.0,
            )
            value = float(interp([[lat, lon]])[0])
            return float(np.clip(value, 0.0, 1.0))

        except Exception:
            pass

        # Nearest-neighbour fallback.
        try:
            sim_lats = np.asarray(simulation_result.lats, dtype=float).ravel()
            sim_lons = np.asarray(simulation_result.lons, dtype=float).ravel()
            risk_flat = np.asarray(simulation_result.flood_risk, dtype=float).ravel()

            distances = np.hypot(sim_lats - lat, sim_lons - lon)
            idx = int(np.argmin(distances))
            return float(np.clip(risk_flat[idx], 0.0, 1.0))

        except Exception:
            return 0.0

    def _check_connectivity(self, G: nx.MultiDiGraph) -> dict:
        """
        Return basic connectivity statistics for the feasible graph.

        Parameters
        ----------
        G : nx.MultiDiGraph

        Returns
        -------
        dict
            Keys: ``num_nodes``, ``num_edges``, ``is_strongly_connected``,
            ``num_strongly_connected_components``,
            ``largest_scc_fraction`` (fraction of nodes in the largest SCC).
        """
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()

        if num_nodes == 0:
            return {
                "num_nodes": 0,
                "num_edges": 0,
                "is_strongly_connected": False,
                "num_strongly_connected_components": 0,
                "largest_scc_fraction": 0.0,
            }

        sccs = list(nx.strongly_connected_components(G))
        num_sccs = len(sccs)
        largest_scc_size = max(len(c) for c in sccs)
        largest_scc_fraction = largest_scc_size / num_nodes if num_nodes > 0 else 0.0

        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "is_strongly_connected": num_sccs == 1,
            "num_strongly_connected_components": num_sccs,
            "largest_scc_fraction": round(largest_scc_fraction, 4),
        }
