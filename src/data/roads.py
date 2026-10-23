"""
Road network data using OSMnx and OpenStreetMap.
Handles graph loading, caching, and road attribute extraction.
"""
import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Callable, Any
from dataclasses import dataclass, field


@dataclass
class RoadSegmentAttributes:
    """
    Attributes for a single road segment (OSM edge).

    Attributes:
        edge_id: OSMnx edge key tuple (u, v, key).
        road_type: OSM highway tag value (e.g. 'primary', 'residential').
        length_m: Physical length of the segment in metres.
        speed_kmh: Design/free-flow speed in km/h.
        travel_time_s: Free-flow travel time in seconds.
        lanes: Number of lanes (1 if unknown).
        surface: OSM surface tag value (e.g. 'asphalt', 'unpaved').
        flood_risk: Flood risk score in [0, 1] assigned by risk model.
        hazard_penalty: Penalty factor applied to travel time during flood events.
        uncertainty: Model uncertainty in [0, 1].
        is_blocked: True when the segment is impassable.
        passability_note: Free-text reason for blockage or degraded passability.
    """
    edge_id: tuple          # (u, v, key)
    road_type: str
    length_m: float
    speed_kmh: float
    travel_time_s: float
    lanes: int
    surface: str
    flood_risk: float = 0.0
    hazard_penalty: float = 0.0
    uncertainty: float = 0.0
    is_blocked: bool = False
    passability_note: str = ""


class RoadNetworkLoader:
    """
    Loads, caches, and enriches OSM road network graphs for flood routing.

    Road graphs are persisted to disk in GraphML format so that subsequent
    runs do not need to re-download them from the OSM Overpass API.
    """

    ROAD_SPEED_DEFAULTS: Dict[str, float] = {
        "motorway": 80.0,
        "trunk": 60.0,
        "primary": 50.0,
        "secondary": 40.0,
        "tertiary": 30.0,
        "residential": 25.0,
        "service": 15.0,
        "unclassified": 25.0,
        "living_street": 10.0,
    }

    def __init__(self, cache_dir: str = "./cache/roads"):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Graph loading and caching
    # ------------------------------------------------------------------

    def _cache_path(self, lat: float, lon: float, radius_km: float) -> Path:
        """Return the GraphML cache file path for the given parameters."""
        key = f"{lat:.3f}_{lon:.3f}_{radius_km}"
        return self._cache_dir / f"road_graph_{key}.graphml"

    def load_graph(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float,
        network_type: str = "drive",
    ) -> nx.MultiDiGraph:
        """
        Load an OSM road graph for the area around (center_lat, center_lon).

        The graph is fetched from a GraphML cache on disk if available;
        otherwise it is downloaded from OSM and saved for future reuse.

        Edge attributes added:
        - ``speed_kph``: speed in km/h (from OSM tags or defaults)
        - ``travel_time``: free-flow travel time in seconds

        Args:
            center_lat: Centre latitude of the area of interest.
            center_lon: Centre longitude of the area of interest.
            radius_km: Radius of the bounding circle in kilometres.
            network_type: OSMnx network type (default ``"drive"``).

        Returns:
            Enriched NetworkX MultiDiGraph.
        """
        cache_file = self._cache_path(center_lat, center_lon, radius_km)

        if cache_file.exists():
            try:
                G = ox.load_graphml(cache_file)
                print(f"[RoadNetworkLoader] Loaded graph from cache: {cache_file}")
                return G
            except Exception as exc:  # noqa: BLE001
                print(f"[RoadNetworkLoader] Cache load failed ({exc}), re-downloading.")

        print(f"[RoadNetworkLoader] Downloading OSM graph for ({center_lat}, {center_lon}) r={radius_km} km …")
        G = ox.graph_from_point(
            (center_lat, center_lon),
            dist=radius_km * 1000,
            network_type=network_type,
        )

        # Add speed and travel-time attributes
        G = self._enrich_speeds(G)

        ox.save_graphml(G, cache_file)
        print(f"[RoadNetworkLoader] Graph cached to {cache_file}")
        return G

    def _enrich_speeds(self, G: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """
        Add ``speed_kph`` and ``travel_time`` attributes to all edges.

        Uses OSMnx's built-in inference first; fills any remaining edges
        using ROAD_SPEED_DEFAULTS.
        """
        try:
            G = ox.add_edge_speeds(G)
            G = ox.add_edge_travel_times(G)
        except Exception as exc:  # noqa: BLE001
            print(f"[RoadNetworkLoader] ox.add_edge_speeds failed: {exc}. Applying defaults.")
            for u, v, k, data in G.edges(keys=True, data=True):
                highway = data.get("highway", "unclassified")
                if isinstance(highway, list):
                    highway = highway[0]
                speed = self.get_road_type_speed(highway)
                length_m = float(data.get("length", 0.0))
                travel_time_s = (length_m / (speed / 3.6)) if speed > 0 else 0.0
                G[u][v][k]["speed_kph"] = speed
                G[u][v][k]["travel_time"] = travel_time_s
            return G

        # Fill any edges still missing speed_kph or travel_time
        for u, v, k, data in G.edges(keys=True, data=True):
            if "speed_kph" not in data or data["speed_kph"] is None or (isinstance(data["speed_kph"], float) and np.isnan(data["speed_kph"])):
                highway = data.get("highway", "unclassified")
                if isinstance(highway, list):
                    highway = highway[0]
                speed = self.get_road_type_speed(highway)
                length_m = float(data.get("length", 0.0))
                travel_time_s = (length_m / (speed / 3.6)) if speed > 0 else 0.0
                G[u][v][k]["speed_kph"] = speed
                G[u][v][k]["travel_time"] = travel_time_s

        return G

    # ------------------------------------------------------------------
    # GeoDataFrame helpers
    # ------------------------------------------------------------------

    def get_edge_gdf(self, G: nx.MultiDiGraph) -> gpd.GeoDataFrame:
        """
        Return graph edges as a GeoDataFrame.

        Columns included (beyond geometry):
        highway, length, speed_kph, travel_time, lanes, surface.

        Args:
            G: Road network graph.

        Returns:
            GeoDataFrame indexed by (u, v, key).
        """
        _, edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
        desired_cols = ["geometry", "highway", "length", "speed_kph", "travel_time", "lanes", "surface"]
        available = [c for c in desired_cols if c in edges.columns]
        gdf = edges[available].copy()

        # Normalise highway column to a single string
        if "highway" in gdf.columns:
            gdf["highway"] = gdf["highway"].apply(
                lambda x: x[0] if isinstance(x, list) else x
            )

        return gdf

    def get_node_gdf(self, G: nx.MultiDiGraph) -> gpd.GeoDataFrame:
        """
        Return graph nodes as a GeoDataFrame with geometry (point) column.

        Args:
            G: Road network graph.

        Returns:
            GeoDataFrame indexed by OSM node ID.
        """
        nodes, _ = ox.graph_to_gdfs(G, nodes=True, edges=False)
        return nodes

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def get_nearest_node(
        self, G: nx.MultiDiGraph, lat: float, lon: float
    ) -> int:
        """
        Find the graph node nearest to the given geographic coordinates.

        Args:
            G: Road network graph.
            lat: Query latitude.
            lon: Query longitude.

        Returns:
            OSM node ID (integer) of the nearest node.
        """
        return ox.distance.nearest_nodes(G, X=lon, Y=lat)

    # ------------------------------------------------------------------
    # Flood risk application
    # ------------------------------------------------------------------

    def apply_flood_risk_to_graph(
        self,
        G: nx.MultiDiGraph,
        flood_risk_fn: Callable[[float, float], float],
        terrain_grid=None,
    ) -> nx.MultiDiGraph:
        """
        Annotate every graph edge with flood risk data.

        For each edge the midpoint coordinates are computed and passed to
        ``flood_risk_fn(lat, lon) -> float`` which should return a risk
        score in [0, 1].

        Edge attributes written:
        - ``flood_risk``: raw risk score in [0, 1].
        - ``is_blocked``: True when risk ≥ 0.9 (road fully inundated).
        - ``hazard_penalty``: multiplicative travel-time penalty in [1, ∞).
          Formula: ``1.0 + risk * 4.0`` (risk=1 → 5× slower).

        Args:
            G: Road network graph to annotate (modified in-place and returned).
            flood_risk_fn: Callable that takes (lat, lon) and returns [0, 1].
            terrain_grid: Optional TerrainGrid (reserved for future use).

        Returns:
            The annotated graph (same object, modified in-place).
        """
        node_data = dict(G.nodes(data=True))

        for u, v, k, data in G.edges(keys=True, data=True):
            u_data = node_data.get(u, {})
            v_data = node_data.get(v, {})

            u_lat = float(u_data.get("y", 0.0))
            u_lon = float(u_data.get("x", 0.0))
            v_lat = float(v_data.get("y", 0.0))
            v_lon = float(v_data.get("x", 0.0))

            mid_lat = (u_lat + v_lat) / 2.0
            mid_lon = (u_lon + v_lon) / 2.0

            try:
                risk = float(flood_risk_fn(mid_lat, mid_lon))
                risk = max(0.0, min(1.0, risk))
            except Exception:  # noqa: BLE001
                risk = 0.0

            is_blocked = risk >= 0.9
            hazard_penalty = 1.0 + risk * 4.0  # [1, 5]

            G[u][v][k]["flood_risk"] = risk
            G[u][v][k]["is_blocked"] = is_blocked
            G[u][v][k]["hazard_penalty"] = hazard_penalty

        return G

    # ------------------------------------------------------------------
    # Speed lookup
    # ------------------------------------------------------------------

    def get_road_type_speed(self, highway_type: str) -> float:
        """
        Return the default free-flow speed (km/h) for an OSM highway type.

        Falls back to the 'unclassified' default for unknown types.

        Args:
            highway_type: Value of the OSM ``highway`` tag.

        Returns:
            Speed in km/h.
        """
        return self.ROAD_SPEED_DEFAULTS.get(
            highway_type,
            self.ROAD_SPEED_DEFAULTS["unclassified"],
        )

    # ------------------------------------------------------------------
    # Connectivity diagnostics
    # ------------------------------------------------------------------

    def graph_connectivity_check(self, G: nx.MultiDiGraph) -> dict:
        """
        Report connectivity statistics for the road graph.

        Returns:
            Dictionary with keys:
            - ``is_strongly_connected``: bool
            - ``num_components``: number of weakly connected components
            - ``largest_component_nodes``: node count of the largest WCC
            - ``total_nodes``: total nodes in G
            - ``total_edges``: total edges in G
        """
        is_sc = nx.is_strongly_connected(G)
        wccs = list(nx.weakly_connected_components(G))
        largest = max((len(c) for c in wccs), default=0)

        return {
            "is_strongly_connected": is_sc,
            "num_components": len(wccs),
            "largest_component_nodes": largest,
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
        }


# ---------------------------------------------------------------------------
# Module-level convenience helper
# ---------------------------------------------------------------------------

_default_loader: Optional[RoadNetworkLoader] = None


def _get_loader() -> RoadNetworkLoader:
    global _default_loader
    if _default_loader is None:
        _default_loader = RoadNetworkLoader()
    return _default_loader


def load_road_network(
    lat: float, lon: float, radius_km: float
) -> nx.MultiDiGraph:
    """
    Convenience function to load the OSM road graph for the given area.

    Delegates to a shared ``RoadNetworkLoader`` instance with the default
    cache directory (``./cache/roads``).

    Args:
        lat: Centre latitude of the area of interest.
        lon: Centre longitude of the area of interest.
        radius_km: Radius of the bounding circle in kilometres.

    Returns:
        Enriched NetworkX MultiDiGraph with speed and travel-time attributes.
    """
    return _get_loader().load_graph(lat, lon, radius_km)
