"""
Flood spread simulation - Level 1 (terrain screening) and Level 2 (simplified runoff accumulation).

Level 1: Uses terrain analysis (elevation, depressions, flow accumulation) to identify
         potentially vulnerable areas. Fast but does not compute actual water depths.

Level 2: Adds SCS-CN runoff estimation and simplified water accumulation/flow routing.
         Produces estimated water depth at each grid cell and flood extent over time.
         
IMPORTANT LIMITATIONS:
- This is NOT a hydrodynamic simulation. It does not solve Saint-Venant equations.
- Water depths are approximate; actual flooding depends on flow dynamics, drainage
  infrastructure, barriers, boundary conditions and many other factors.
- Use results for relative comparison and screening, not precise engineering decisions.
- For rigorous flood modelling, use HEC-RAS or similar hydraulic simulation software.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timedelta
from scipy import ndimage

from .runoff import SCSRunoffModel, LandCoverType, AntecedentMoisture

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LEVEL1_LIMITATIONS = (
    "Level 1 (terrain screening): identifies potentially vulnerable cells based on "
    "elevation, depressions, and flow accumulation. Water depths are proxies, NOT "
    "hydraulically simulated values. No Saint-Venant equations are solved. "
    "For rigorous flood modelling use HEC-RAS or similar software."
)

_LEVEL2_LIMITATIONS = (
    "Level 2 (simplified runoff accumulation): uses SCS-CN runoff estimation and "
    "a D8 flow-routing approximation. Water depths are indicative only and have NOT "
    "been calibrated against observed flood events. Drainage infrastructure, embankments, "
    "and boundary conditions are not represented. Do not use for engineering design."
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SimulationScenario:
    """Defines the input parameters for a flood simulation run."""
    scenario_id: str
    name: str
    center_lat: float
    center_lon: float
    radius_km: float
    rainfall_mm_hr: float
    duration_hours: float
    land_cover: str = "suburban"
    created_at: datetime = field(default_factory=datetime.utcnow)
    description: str = ""


@dataclass
class SimulationTimeStep:
    """Snapshot of the simulated flood state at a single time step."""
    step_index: int
    elapsed_minutes: int
    water_depth_m: np.ndarray        # 2-D array [rows, cols]
    flood_extent_mask: np.ndarray    # boolean 2-D array
    flooded_area_km2: float
    max_depth_m: float
    mean_depth_m: float


@dataclass
class SimulationResult:
    """
    Full output of a flood simulation run.

    Arrays are 2-D with shape (n_rows, n_cols) corresponding to
    grid_lats (1-D, length n_rows) and grid_lons (1-D, length n_cols).
    """
    scenario: SimulationScenario
    grid_lats: np.ndarray
    grid_lons: np.ndarray
    time_steps: List[SimulationTimeStep]
    final_flood_extent_mask: np.ndarray
    flood_arrival_time_minutes: np.ndarray  # -1 = never flooded
    max_depth_m: np.ndarray                 # max depth per cell over all time steps
    simulation_level: int                    # 1 or 2
    limitations: str
    completed_at: datetime = field(default_factory=datetime.utcnow)

    def summary(self) -> dict:
        """
        Returns a concise summary dictionary of the simulation output.

        Keys
        ----
        total_area_km2     : Total simulated area in km².
        flooded_area_km2   : Final flooded area in km².
        flood_fraction     : flooded_area / total_area (0-1).
        max_depth_m        : Maximum water depth across the entire grid.
        simulation_level   : 1 (terrain screening) or 2 (runoff accumulation).
        duration_hours     : Storm duration from the scenario.
        time_steps         : Number of recorded time steps.
        """
        # Approximate cell area
        if len(self.grid_lats) > 1 and len(self.grid_lons) > 1:
            dlat = abs(float(self.grid_lats[1] - self.grid_lats[0]))
            dlon = abs(float(self.grid_lons[1] - self.grid_lons[0]))
            # 1° lat ≈ 111 km; use mean lat for lon
            mean_lat = float(np.mean(self.grid_lats))
            cell_area_km2 = (dlat * 111.0) * (dlon * 111.0 * np.cos(np.radians(mean_lat)))
            total_cells = self.grid_lats.size * self.grid_lons.size
            total_area_km2 = float(total_cells * cell_area_km2)
        else:
            total_area_km2 = float(np.pi * self.scenario.radius_km ** 2)

        flooded_cells = int(np.sum(self.final_flood_extent_mask))
        total_cells_grid = self.final_flood_extent_mask.size
        flood_fraction = flooded_cells / total_cells_grid if total_cells_grid > 0 else 0.0
        flooded_area_km2 = flood_fraction * total_area_km2

        return {
            "total_area_km2": round(total_area_km2, 3),
            "flooded_area_km2": round(flooded_area_km2, 3),
            "flood_fraction": round(flood_fraction, 4),
            "max_depth_m": round(float(np.max(self.max_depth_m)), 3),
            "simulation_level": self.simulation_level,
            "duration_hours": self.scenario.duration_hours,
            "time_steps": len(self.time_steps),
        }


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class FloodSimulator:
    """
    Terrain-based flood spread simulator.

    Level 1 – terrain screening only (fast).
    Level 2 – SCS-CN runoff + simplified D8 routing (slower, more detail).
    """

    # D8 direction offsets: (dy, dx) for 8 neighbours
    _D8_OFFSETS: List[Tuple[int, int]] = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),           ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1),
    ]

    def __init__(self, terrain_grid=None, runoff_model: SCSRunoffModel = None) -> None:
        """
        Parameters
        ----------
        terrain_grid  : Optional pre-loaded terrain grid object.
        runoff_model  : Optional SCSRunoffModel instance. A default instance
                        is created if not provided.
        """
        self.terrain_grid = terrain_grid
        self.runoff_model = runoff_model if runoff_model is not None else SCSRunoffModel()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_synthetic_grid(
        center_lat: float,
        center_lon: float,
        radius_km: float,
        grid_size: int = 30,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build a synthetic elevation grid centred on (center_lat, center_lon).

        Returns (grid_lats, grid_lons, elevation_m) where elevation is a
        smoothly varying synthetic landscape used when no real DEM is available.
        """
        lat_span = radius_km / 111.0        # degrees
        lon_span = radius_km / (111.0 * np.cos(np.radians(center_lat)))

        grid_lats = np.linspace(center_lat - lat_span, center_lat + lat_span, grid_size)
        grid_lons = np.linspace(center_lon - lon_span, center_lon + lon_span, grid_size)

        # Synthetic elevation: gentle bowl shape with random noise
        rng = np.random.default_rng(seed=int(abs(center_lat * 1000 + center_lon * 100)) % (2**31))
        y = np.linspace(-1, 1, grid_size)
        x = np.linspace(-1, 1, grid_size)
        xx, yy = np.meshgrid(x, y)

        # Bowl = low in centre, higher at edges; add noise
        elevation = 15.0 * (xx**2 + yy**2) + rng.normal(0, 1.5, (grid_size, grid_size))
        elevation = np.clip(elevation, 0.0, None)

        return grid_lats, grid_lons, elevation

    @staticmethod
    def _extract_terrain_arrays(
        terrain_grid,
        center_lat: float,
        center_lon: float,
        radius_km: float,
        grid_size: int = 30,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract or synthesise elevation, depression, and flow_accumulation arrays.

        Returns (grid_lats, grid_lons, elevation_m, depressions, flow_accum)
        """
        if terrain_grid is not None:
            # Attempt to read arrays from the terrain_grid object
            try:
                elevations = np.array(terrain_grid.elevations, dtype=float)
                grid_lats = np.array(getattr(terrain_grid, "lats",
                                              np.linspace(center_lat - 0.05, center_lat + 0.05, elevations.shape[0])))
                grid_lons = np.array(getattr(terrain_grid, "lons",
                                              np.linspace(center_lon - 0.05, center_lon + 0.05, elevations.shape[1])))

                depressions = np.array(
                    getattr(terrain_grid, "depressions", np.zeros_like(elevations, dtype=bool)),
                    dtype=bool,
                )
                flow_accum = np.array(
                    getattr(terrain_grid, "flow_accumulation", np.zeros_like(elevations)),
                    dtype=float,
                )
                return grid_lats, grid_lons, elevations, depressions, flow_accum
            except (AttributeError, TypeError):
                pass   # fall through to synthetic grid

        grid_lats, grid_lons, elevations = FloodSimulator._build_synthetic_grid(
            center_lat, center_lon, radius_km, grid_size
        )
        # Derive synthetic depressions and flow accumulation
        # Depression: local minima in elevation
        local_min = ndimage.minimum_filter(elevations, size=3) == elevations
        depressions = local_min & (elevations < np.percentile(elevations, 25))

        # Flow accumulation: approximate by Gaussian-blurred inverse elevation
        inv_elev = np.max(elevations) - elevations
        flow_accum = ndimage.gaussian_filter(inv_elev, sigma=2.0) * 10.0

        return grid_lats, grid_lons, elevations, depressions, flow_accum

    # ------------------------------------------------------------------ #
    # D8 flow routing
    # ------------------------------------------------------------------ #

    def _d8_flow_direction(self, elevation: np.ndarray) -> np.ndarray:
        """
        Compute D8 flow direction for each cell.

        Each cell is assigned to its steepest-descent 8-neighbour.
        Border cells flow inward (clamped).

        Returns
        -------
        np.ndarray of shape (rows, cols, 2)
            flow_dirs[r, c] = (dy, dx) offset to downstream neighbour.
            (0, 0) if the cell is a local minimum (sink).
        """
        rows, cols = elevation.shape
        flow_dirs = np.zeros((rows, cols, 2), dtype=int)

        for r in range(rows):
            for c in range(cols):
                best_drop = 0.0
                best_dy, best_dx = 0, 0
                for dy, dx in self._D8_OFFSETS:
                    nr, nc = r + dy, c + dx
                    if 0 <= nr < rows and 0 <= nc < cols:
                        drop = elevation[r, c] - elevation[nr, nc]
                        if drop > best_drop:
                            best_drop = drop
                            best_dy, best_dx = dy, dx
                flow_dirs[r, c, 0] = best_dy
                flow_dirs[r, c, 1] = best_dx

        return flow_dirs

    def _route_water_step(
        self,
        water_depth: np.ndarray,
        elevation: np.ndarray,
        flow_dirs: np.ndarray,
        dt_hr: float,
        runoff_m: float,
    ) -> np.ndarray:
        """
        Advance water depth by one time step.

        Algorithm:
          1. Add runoff_m of water uniformly to every cell.
          2. Each cell transfers a fraction of its water to its downstream
             neighbour (D8 routing). Transfer fraction is proportional to
             slope (steeper = faster) and time step.
          3. Cells with no downstream neighbour retain all water.

        Parameters
        ----------
        water_depth : np.ndarray – current water depth (m), shape (rows, cols)
        elevation   : np.ndarray – terrain elevation (m)
        flow_dirs   : np.ndarray – D8 directions, shape (rows, cols, 2)
        dt_hr       : float      – time step duration in hours
        runoff_m    : float      – runoff depth added each step (m)

        Returns
        -------
        np.ndarray
            Updated water_depth array.
        """
        rows, cols = water_depth.shape
        new_depth = water_depth.copy()

        # Add runoff to every cell
        new_depth += runoff_m

        transfer = np.zeros_like(water_depth)

        for r in range(rows):
            for c in range(cols):
                dy = int(flow_dirs[r, c, 0])
                dx = int(flow_dirs[r, c, 1])
                if dy == 0 and dx == 0:
                    continue   # sink / local minimum – water stays

                nr, nc = r + dy, c + dx
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue

                # Slope between cell and downstream neighbour
                distance = np.sqrt(dy**2 + dx**2)   # 1.0 or √2 cell units
                slope = max((elevation[r, c] - elevation[nr, nc]) / distance, 0.0)

                # Transfer coefficient: calibrated so slope=1 → ~50% per hour
                transfer_fraction = min(slope * 0.5 * dt_hr, 0.9)
                amount = new_depth[r, c] * transfer_fraction
                transfer[r, c] -= amount
                transfer[nr, nc] += amount

        new_depth += transfer
        new_depth = np.maximum(new_depth, 0.0)
        return new_depth

    # ------------------------------------------------------------------ #
    # Flood arrival time
    # ------------------------------------------------------------------ #

    def _compute_flood_arrival_time(
        self,
        time_steps: List[SimulationTimeStep],
        threshold_m: float = 0.1,
    ) -> np.ndarray:
        """
        Determine the elapsed time (minutes) when each cell first floods.

        Parameters
        ----------
        time_steps  : list of SimulationTimeStep
        threshold_m : float – water depth threshold for "flooded" (default 0.1 m)

        Returns
        -------
        np.ndarray (int32)
            Minutes until flooded for each cell. -1 = never flooded.
        """
        if not time_steps:
            return np.full((1, 1), -1, dtype=np.int32)

        shape = time_steps[0].water_depth_m.shape
        arrival = np.full(shape, -1, dtype=np.int32)

        for ts in time_steps:
            newly_flooded = (ts.water_depth_m >= threshold_m) & (arrival == -1)
            arrival[newly_flooded] = ts.elapsed_minutes

        return arrival

    # ------------------------------------------------------------------ #
    # Road hazard interpolation
    # ------------------------------------------------------------------ #

    def get_road_hazard_at_points(
        self,
        result: SimulationResult,
        road_midpoints: List[Tuple[float, float]],
    ) -> List[float]:
        """
        Interpolate maximum water depth at road midpoints from the simulation grid.

        Uses bilinear interpolation (nearest-neighbour fallback if out of bounds).

        Parameters
        ----------
        result        : SimulationResult
        road_midpoints: list of (lat, lon) tuples

        Returns
        -------
        list of float
            Water depth in metres at each road midpoint.
        """
        depths: List[float] = []
        max_depth_grid = result.max_depth_m
        lats = result.grid_lats
        lons = result.grid_lons

        lat_min, lat_max = float(lats.min()), float(lats.max())
        lon_min, lon_max = float(lons.min()), float(lons.max())

        n_rows, n_cols = max_depth_grid.shape

        for lat, lon in road_midpoints:
            # Check bounds
            if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
                depths.append(0.0)
                continue

            # Fractional row/col index
            row_f = (lat - lat_min) / (lat_max - lat_min) * (n_rows - 1) if n_rows > 1 else 0.0
            col_f = (lon - lon_min) / (lon_max - lon_min) * (n_cols - 1) if n_cols > 1 else 0.0

            r0, c0 = int(np.floor(row_f)), int(np.floor(col_f))
            r1, c1 = min(r0 + 1, n_rows - 1), min(c0 + 1, n_cols - 1)
            dr, dc = row_f - r0, col_f - c0

            # Bilinear interpolation
            depth = (
                max_depth_grid[r0, c0] * (1 - dr) * (1 - dc)
                + max_depth_grid[r1, c0] * dr * (1 - dc)
                + max_depth_grid[r0, c1] * (1 - dr) * dc
                + max_depth_grid[r1, c1] * dr * dc
            )
            depths.append(float(depth))

        return depths

    # ------------------------------------------------------------------ #
    # Level 1
    # ------------------------------------------------------------------ #

    def run_level1(
        self,
        scenario: SimulationScenario,
        terrain_grid,
    ) -> SimulationResult:
        """
        Level 1 flood screening using terrain attributes only.

        Identifies at-risk cells as those that are:
          - Topographic depressions, OR
          - High flow-accumulation cells (> 50th percentile), OR
          - Below the 25th-percentile elevation

        The water-depth proxy grows exponentially with time (simulating
        progressive filling) and saturates by the end of the storm.

        Parameters
        ----------
        scenario     : SimulationScenario
        terrain_grid : Terrain data object (or None for synthetic grid).

        Returns
        -------
        SimulationResult with simulation_level = 1
        """
        grid_lats, grid_lons, elevations, depressions, flow_accum = (
            self._extract_terrain_arrays(
                terrain_grid,
                scenario.center_lat,
                scenario.center_lon,
                scenario.radius_km,
            )
        )

        low_elevation_thresh = float(np.percentile(elevations, 25))
        flow_thresh = float(np.percentile(flow_accum, 50))

        at_risk = depressions | (flow_accum > flow_thresh) | (elevations <= low_elevation_thresh)

        # Normalised flow accumulation as depth proxy
        flow_norm = np.where(at_risk, flow_accum / (flow_accum.max() + 1e-9), 0.0)

        # Time steps every 30 minutes
        step_minutes = 30
        n_steps = max(1, int(scenario.duration_hours * 60 / step_minutes))
        time_steps: List[SimulationTimeStep] = []
        flood_depth_thresh = 0.1   # m

        for i in range(n_steps):
            elapsed = (i + 1) * step_minutes
            # Exponential growth towards saturation
            fraction = 1.0 - np.exp(-3.0 * elapsed / (scenario.duration_hours * 60))
            depth_proxy = flow_norm * (scenario.rainfall_mm_hr / 50.0) * fraction
            depth_proxy = np.maximum(depth_proxy, 0.0)

            flood_mask = depth_proxy >= flood_depth_thresh
            flooded_cells = int(np.sum(flood_mask))
            total_cells = depth_proxy.size
            cell_area_km2 = (np.pi * scenario.radius_km ** 2) / total_cells

            time_steps.append(SimulationTimeStep(
                step_index=i,
                elapsed_minutes=elapsed,
                water_depth_m=depth_proxy.copy(),
                flood_extent_mask=flood_mask.copy(),
                flooded_area_km2=float(flooded_cells * cell_area_km2),
                max_depth_m=float(np.max(depth_proxy)),
                mean_depth_m=float(np.mean(depth_proxy[flood_mask])) if flooded_cells > 0 else 0.0,
            ))

        final_mask = time_steps[-1].flood_extent_mask if time_steps else np.zeros_like(at_risk)
        max_depth_grid = np.maximum.reduce([ts.water_depth_m for ts in time_steps]) if time_steps else np.zeros_like(elevations)
        arrival_times = self._compute_flood_arrival_time(time_steps, threshold_m=flood_depth_thresh)

        return SimulationResult(
            scenario=scenario,
            grid_lats=grid_lats,
            grid_lons=grid_lons,
            time_steps=time_steps,
            final_flood_extent_mask=final_mask,
            flood_arrival_time_minutes=arrival_times,
            max_depth_m=max_depth_grid,
            simulation_level=1,
            limitations=_LEVEL1_LIMITATIONS,
        )

    # ------------------------------------------------------------------ #
    # Level 2
    # ------------------------------------------------------------------ #

    def run_level2(
        self,
        scenario: SimulationScenario,
        terrain_grid,
    ) -> SimulationResult:
        """
        Level 2 simplified runoff accumulation and D8 flow routing.

        Steps
        -----
        1. Resolve land cover and antecedent moisture from scenario.
        2. Use SCS-CN to compute runoff depth per time step (dt = 0.5 hr).
        3. Compute D8 flow directions from the DEM.
        4. For each time step: add runoff → route water → record state.
        5. Flood extent = cells with accumulated depth ≥ 0.1 m.

        Parameters
        ----------
        scenario     : SimulationScenario
        terrain_grid : Terrain data object (or None for synthetic grid).

        Returns
        -------
        SimulationResult with simulation_level = 2
        """
        grid_lats, grid_lons, elevations, depressions, flow_accum = (
            self._extract_terrain_arrays(
                terrain_grid,
                scenario.center_lat,
                scenario.center_lon,
                scenario.radius_km,
            )
        )

        # Resolve land cover type
        lc_map = {lct.value: lct for lct in LandCoverType}
        land_cover = lc_map.get(scenario.land_cover.lower(), LandCoverType.SUBURBAN)

        # Determine AMC from scenario duration as proxy for antecedent wetness
        # (conservative: assume NORMAL for < 6 h storms, WET for longer)
        amc = AntecedentMoisture.WET if scenario.duration_hours >= 6 else AntecedentMoisture.NORMAL

        # Compute D8 flow directions (expensive; done once)
        flow_dirs = self._d8_flow_direction(elevations)

        # Time stepping (every 30 min)
        dt_hr = 0.5
        n_steps = max(1, int(scenario.duration_hours / dt_hr))
        flood_depth_thresh = 0.1   # m

        cell_area_km2 = (np.pi * scenario.radius_km ** 2) / elevations.size

        water_depth = np.zeros_like(elevations)
        time_steps: List[SimulationTimeStep] = []

        for i in range(n_steps):
            elapsed = int((i + 1) * dt_hr * 60)
            rainfall_for_step = scenario.rainfall_mm_hr * dt_hr   # mm per step

            runoff_est = self.runoff_model.compute_runoff(rainfall_for_step, land_cover, amc)
            runoff_m = runoff_est.runoff_mm / 1000.0

            water_depth = self._route_water_step(
                water_depth, elevations, flow_dirs, dt_hr, runoff_m
            )

            flood_mask = water_depth >= flood_depth_thresh
            flooded_cells = int(np.sum(flood_mask))

            time_steps.append(SimulationTimeStep(
                step_index=i,
                elapsed_minutes=elapsed,
                water_depth_m=water_depth.copy(),
                flood_extent_mask=flood_mask.copy(),
                flooded_area_km2=float(flooded_cells * cell_area_km2),
                max_depth_m=float(np.max(water_depth)),
                mean_depth_m=float(np.mean(water_depth[flood_mask])) if flooded_cells > 0 else 0.0,
            ))

        final_mask = time_steps[-1].flood_extent_mask if time_steps else np.zeros(elevations.shape, dtype=bool)
        max_depth_grid = np.maximum.reduce([ts.water_depth_m for ts in time_steps]) if time_steps else np.zeros_like(elevations)
        arrival_times = self._compute_flood_arrival_time(time_steps, threshold_m=flood_depth_thresh)

        return SimulationResult(
            scenario=scenario,
            grid_lats=grid_lats,
            grid_lons=grid_lons,
            time_steps=time_steps,
            final_flood_extent_mask=final_mask,
            flood_arrival_time_minutes=arrival_times,
            max_depth_m=max_depth_grid,
            simulation_level=2,
            limitations=_LEVEL2_LIMITATIONS,
        )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def run_simulation(
    scenario: SimulationScenario,
    terrain_grid,
    level: int = 2,
) -> SimulationResult:
    """
    Convenience wrapper that runs a flood simulation at the requested level.

    Parameters
    ----------
    scenario     : SimulationScenario – input parameters.
    terrain_grid : Terrain data object or None (uses synthetic DEM).
    level        : int – 1 for terrain screening, 2 for runoff simulation.

    Returns
    -------
    SimulationResult
    """
    simulator = FloodSimulator()
    if level == 1:
        return simulator.run_level1(scenario, terrain_grid)
    elif level == 2:
        return simulator.run_level2(scenario, terrain_grid)
    else:
        raise ValueError(f"Unsupported simulation level: {level}. Use 1 or 2.")
