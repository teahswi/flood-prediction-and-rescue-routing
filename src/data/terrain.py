"""
Terrain data: elevation, slope, flow direction from SRTM via Open-Topo-Data API.
API: https://www.opentopodata.org/
"""
import numpy as np
import pandas as pd
import requests
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple, Optional


@dataclass
class TerrainPoint:
    """Terrain attributes at a single geographic point."""
    lat: float
    lon: float
    elevation_m: float
    slope_deg: float = 0.0
    aspect_deg: float = 0.0      # flow direction (degrees from north, clockwise)
    is_depression: bool = False  # local low point
    flow_accumulation: float = 0.0


@dataclass
class TerrainGrid:
    """
    A 2-D raster of terrain attributes covering a rectangular region.

    Attributes:
        lats: 1-D array of row latitudes (length = rows).
        lons: 1-D array of column longitudes (length = cols).
        elevations: 2-D array of elevation values, shape (rows, cols).
        slopes: 2-D array of slope in degrees, shape (rows, cols).
        aspects: 2-D array of aspect (flow direction) in degrees, shape (rows, cols).
        depressions: 2-D boolean mask of local depressions, shape (rows, cols).
        flow_accumulation: 2-D array of upstream cell counts, shape (rows, cols).
        cell_size_deg: Grid spacing in decimal degrees.
        fetched_at: ISO-8601 timestamp of when the grid was created.
    """
    lats: np.ndarray
    lons: np.ndarray
    elevations: np.ndarray
    slopes: np.ndarray
    aspects: np.ndarray
    depressions: np.ndarray
    flow_accumulation: np.ndarray
    cell_size_deg: float
    fetched_at: str

    def terrain_risk_score(self) -> np.ndarray:
        """
        Compute a terrain-based flood risk score in [0, 1] for every grid cell.

        Formula:
            risk = 0.4 * norm(flow_accumulation)
                 + 0.4 * (1 - norm(elevation))
                 + 0.2 * depressions

        where norm(x) = (x - min) / (max - min), clamped to [0, 1].

        Returns:
            2-D ndarray of risk scores, same shape as elevations.
        """
        def _norm(arr: np.ndarray) -> np.ndarray:
            a_min, a_max = arr.min(), arr.max()
            if a_max == a_min:
                return np.zeros_like(arr, dtype=float)
            return (arr - a_min) / (a_max - a_min)

        norm_flow = _norm(self.flow_accumulation.astype(float))
        norm_elev = _norm(self.elevations.astype(float))
        dep_float = self.depressions.astype(float)

        risk = 0.4 * norm_flow + 0.4 * (1.0 - norm_elev) + 0.2 * dep_float
        return np.clip(risk, 0.0, 1.0)


class TerrainClient:
    """
    Client for the Open-Topo-Data SRTM-30 m elevation API.
    Builds terrain grids with slope, aspect, depression detection, and
    a simple flow-accumulation approximation.
    """

    OPEN_TOPO_URL = "https://api.opentopodata.org/v1/srtm30m"
    _MAX_BATCH = 100  # API hard limit per request

    def __init__(self, timeout_seconds: int = 30):
        self.timeout = timeout_seconds
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    # ------------------------------------------------------------------
    # Low-level elevation fetch
    # ------------------------------------------------------------------

    def fetch_elevation_points(
        self, locations: List[Tuple[float, float]]
    ) -> List[float]:
        """
        Fetch elevations for a list of (lat, lon) tuples.

        Automatically batches requests to respect the 100-point API limit.

        Args:
            locations: List of (latitude, longitude) pairs.

        Returns:
            List of elevation values in metres. NaN is returned for any
            point that could not be resolved.
        """
        elevations: List[float] = []
        for batch_start in range(0, len(locations), self._MAX_BATCH):
            batch = locations[batch_start: batch_start + self._MAX_BATCH]
            elevations.extend(self._fetch_batch(batch))
        return elevations

    def _fetch_batch(self, batch: List[Tuple[float, float]]) -> List[float]:
        """Fetch one batch (≤100 points) from Open-Topo-Data."""
        loc_str = "|".join(f"{lat},{lon}" for lat, lon in batch)
        try:
            resp = self._session.get(
                self.OPEN_TOPO_URL,
                params={"locations": loc_str},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            out = []
            for r in results:
                elev = r.get("elevation")
                out.append(float(elev) if elev is not None else np.nan)
            # Pad with NaN if the API returned fewer results than expected
            while len(out) < len(batch):
                out.append(np.nan)
            return out
        except Exception as exc:  # noqa: BLE001
            print(f"[TerrainClient] Batch elevation request failed: {exc}")
            return [np.nan] * len(batch)

    # ------------------------------------------------------------------
    # Grid construction
    # ------------------------------------------------------------------

    def build_terrain_grid(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float,
        grid_points: int = 20,
    ) -> TerrainGrid:
        """
        Build a terrain grid centred on (center_lat, center_lon).

        The grid covers a square bounding box of side 2*radius_km (approx.)
        and contains grid_points × grid_points cells.

        Steps:
        1. Generate evenly-spaced lat/lon arrays.
        2. Batch-fetch SRTM elevations for all grid_points² points.
        3. Compute slope (degrees) and aspect from the elevation gradient.
        4. Detect depressions (cells lower than all 8 neighbours).
        5. Estimate flow accumulation via iterative drainage to lowest neighbour.

        Args:
            center_lat: Centre latitude.
            center_lon: Centre longitude.
            radius_km: Half-size of the bounding box in kilometres.
            grid_points: Number of points along each axis.

        Returns:
            TerrainGrid instance with all derived attributes populated.
        """
        # ------------------------------------------------------------------
        # 1. Generate grid coordinates
        # ------------------------------------------------------------------
        deg_per_km_lat = 1.0 / 111.0
        deg_per_km_lon = 1.0 / (111.0 * np.cos(np.radians(center_lat)))

        delta_lat = radius_km * deg_per_km_lat
        delta_lon = radius_km * deg_per_km_lon

        lats = np.linspace(center_lat - delta_lat, center_lat + delta_lat, grid_points)
        lons = np.linspace(center_lon - delta_lon, center_lon + delta_lon, grid_points)
        cell_size_deg = float(lats[1] - lats[0]) if grid_points > 1 else delta_lat * 2

        # Flatten into a list of (lat, lon) pairs, row-major
        locations = [(float(lat), float(lon)) for lat in lats for lon in lons]

        # ------------------------------------------------------------------
        # 2. Fetch elevations
        # ------------------------------------------------------------------
        raw_elevations = self.fetch_elevation_points(locations)
        elev_2d = np.array(raw_elevations, dtype=float).reshape(grid_points, grid_points)

        # Replace NaN with nearest-neighbor fill (simple column-then-row forward fill)
        elev_2d = self._fill_nan_elevation(elev_2d)

        # ------------------------------------------------------------------
        # 3. Slope and aspect from gradient
        # ------------------------------------------------------------------
        # Physical cell size in metres for gradient scaling
        cell_size_m = cell_size_deg * 111_000.0  # approximate

        # np.gradient returns [d/drow, d/dcol] i.e. [d/dlat, d/dlon]
        grad_lat, grad_lon = np.gradient(elev_2d, cell_size_m, cell_size_m)

        slopes = np.degrees(np.arctan(np.sqrt(grad_lat**2 + grad_lon**2)))
        # Aspect: direction of steepest descent (flow direction), 0=N, clockwise
        aspects = np.degrees(np.arctan2(-grad_lon, -grad_lat)) % 360.0

        # ------------------------------------------------------------------
        # 4. Depression detection
        # ------------------------------------------------------------------
        depressions = self._detect_depressions(elev_2d)

        # ------------------------------------------------------------------
        # 5. Flow accumulation
        # ------------------------------------------------------------------
        flow_acc = self._compute_flow_accumulation(elev_2d)

        return TerrainGrid(
            lats=lats,
            lons=lons,
            elevations=elev_2d,
            slopes=slopes,
            aspects=aspects,
            depressions=depressions,
            flow_accumulation=flow_acc,
            cell_size_deg=cell_size_deg,
            fetched_at=datetime.utcnow().isoformat(),
        )

    # ------------------------------------------------------------------
    # Point-level convenience
    # ------------------------------------------------------------------

    def get_terrain_risk_at_point(
        self, lat: float, lon: float, radius_km: float = 2.0
    ) -> float:
        """
        Return a terrain flood-risk score in [0, 1] for the given point.

        Uses a small 5×5 terrain grid centred on the point and returns
        the risk score at the central cell.

        Args:
            lat: Latitude.
            lon: Longitude.
            radius_km: Half-size of the local terrain analysis window.

        Returns:
            Risk score between 0.0 (low) and 1.0 (high).
        """
        grid = self.build_terrain_grid(lat, lon, radius_km=radius_km, grid_points=5)
        risk = grid.terrain_risk_score()
        centre = risk.shape[0] // 2
        return float(risk[centre, centre])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_nan_elevation(arr: np.ndarray) -> np.ndarray:
        """Replace NaN values in a 2-D array using forward/backward fill."""
        filled = arr.copy()
        rows, cols = filled.shape
        # Forward-fill along columns
        for r in range(rows):
            for c in range(1, cols):
                if np.isnan(filled[r, c]) and not np.isnan(filled[r, c - 1]):
                    filled[r, c] = filled[r, c - 1]
        # Backward-fill along columns
        for r in range(rows):
            for c in range(cols - 2, -1, -1):
                if np.isnan(filled[r, c]) and not np.isnan(filled[r, c + 1]):
                    filled[r, c] = filled[r, c + 1]
        # Forward-fill along rows
        for c in range(cols):
            for r in range(1, rows):
                if np.isnan(filled[r, c]) and not np.isnan(filled[r - 1, c]):
                    filled[r, c] = filled[r - 1, c]
        # Replace any remaining NaN with 0
        filled = np.where(np.isnan(filled), 0.0, filled)
        return filled

    @staticmethod
    def _detect_depressions(elev: np.ndarray) -> np.ndarray:
        """
        Identify cells that are lower than all 8 neighbours (D8 sinks).

        Returns a boolean array of the same shape as elev.
        """
        rows, cols = elev.shape
        dep = np.zeros((rows, cols), dtype=bool)
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                neighbours = elev[r - 1: r + 2, c - 1: c + 2].copy()
                # Exclude the centre cell itself
                centre_val = neighbours[1, 1]
                mask = np.ones((3, 3), dtype=bool)
                mask[1, 1] = False
                if np.all(centre_val <= neighbours[mask]):
                    dep[r, c] = True
        return dep

    @staticmethod
    def _compute_flow_accumulation(elev: np.ndarray) -> np.ndarray:
        """
        Approximate D8 flow accumulation by routing each cell to its lowest neighbour.

        Each cell contributes 1 unit to every cell downstream of it along the
        steepest-descent path.  The algorithm iterates from highest to lowest
        elevation to propagate counts.

        Returns:
            2-D ndarray of upstream cell counts (integers stored as float).
        """
        rows, cols = elev.shape
        acc = np.zeros((rows, cols), dtype=float)

        # D8 neighbour offsets: (dr, dc)
        d8 = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),          (0, 1),
              (1, -1),  (1, 0), (1, 1)]

        # Process cells from highest to lowest elevation
        flat_order = np.argsort(elev.ravel())[::-1]
        for flat_idx in flat_order:
            r, c = divmod(int(flat_idx), cols)
            best_dr, best_dc = 0, 0
            best_drop = 0.0
            for dr, dc in d8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    drop = elev[r, c] - elev[nr, nc]
                    if drop > best_drop:
                        best_drop = drop
                        best_dr, best_dc = dr, dc
            if best_drop > 0:
                acc[r + best_dr, c + best_dc] += acc[r, c] + 1.0

        return acc


# ---------------------------------------------------------------------------
# Module-level convenience helper
# ---------------------------------------------------------------------------

_terrain_cache: dict = {}
_TERRAIN_CACHE_TTL = 86_400  # 24 hours in seconds


def get_terrain_grid(
    center_lat: float,
    center_lon: float,
    radius_km: float,
    use_cache: bool = True,
) -> TerrainGrid:
    """
    Convenience function to build (or retrieve from cache) a TerrainGrid.

    Results are cached in-process for 24 hours because terrain data changes
    very infrequently.

    Args:
        center_lat: Centre latitude.
        center_lon: Centre longitude.
        radius_km: Half-size of the analysis bounding box in kilometres.
        use_cache: If True, return a cached grid when available.

    Returns:
        TerrainGrid populated with elevations, slopes, aspects, depressions,
        and flow accumulation.
    """
    from datetime import datetime as _dt

    cache_key = f"terrain_{center_lat:.3f}_{center_lon:.3f}_{radius_km:.1f}"

    if use_cache and cache_key in _terrain_cache:
        grid, cached_at = _terrain_cache[cache_key]
        age = (_dt.utcnow() - cached_at).total_seconds()
        if age < _TERRAIN_CACHE_TTL:
            return grid

    client = TerrainClient()
    grid = client.build_terrain_grid(center_lat, center_lon, radius_km)

    if use_cache:
        from datetime import datetime as _dt2
        _terrain_cache[cache_key] = (grid, _dt2.utcnow())

    return grid
