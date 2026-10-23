"""
Rainfall-runoff estimation using the SCS Curve Number (CN) method.

The SCS-CN method estimates direct runoff volume from rainfall depth
based on land cover and antecedent soil moisture conditions.

Reference: USDA-NRCS National Engineering Handbook, Part 630, Chapter 10.

LIMITATIONS:
- CN values are approximations; local calibration improves accuracy
- Method is calibrated for storm events, not continuous simulation
- Does not account for dynamic infiltration rates
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class LandCoverType(Enum):
    """Primary land-cover categories used in CN lookup."""
    URBAN = "urban"
    SUBURBAN = "suburban"
    RURAL = "rural"
    FOREST = "forest"
    WATER = "water"
    BARREN = "barren"


class AntecedentMoisture(Enum):
    """
    Antecedent Moisture Condition (AMC) classes.

    AMC I   (DRY)    : < 35 mm rainfall in prior 5 days
    AMC II  (NORMAL) : 35–53 mm rainfall in prior 5 days
    AMC III (WET)    : > 53 mm rainfall in prior 5 days
    """
    DRY = 1
    NORMAL = 2
    WET = 3


@dataclass
class RunoffEstimate:
    """
    Result of a single SCS-CN runoff calculation.

    All depths are in millimetres unless otherwise noted.
    """
    rainfall_mm: float
    runoff_mm: float
    retention_mm: float          # water retained in soil (P - Q)
    runoff_coefficient: float    # dimensionless Q/P ratio (0 if P == 0)
    cn_used: int
    land_cover: str
    amc: str
    limitations: str = ""


class SCSRunoffModel:
    """
    SCS Curve Number runoff model.

    Lookup table sourced from USDA-NRCS NEH Part 630, Chapter 9 (Table 9-1)
    and adjusted for AMC conditions using standard conversion equations.

    CN adjustments for AMC I and III use the empirical Chow et al. (1988)
    equations applied to the standard AMC II values.
    """

    # Base CN table (AMC II) and adjusted values for AMC I and III
    # Format: LandCoverType -> {AntecedentMoisture -> CN}
    CN_TABLE = {
        LandCoverType.URBAN: {
            AntecedentMoisture.DRY: 77,
            AntecedentMoisture.NORMAL: 90,
            AntecedentMoisture.WET: 95,
        },
        LandCoverType.SUBURBAN: {
            AntecedentMoisture.DRY: 61,
            AntecedentMoisture.NORMAL: 77,
            AntecedentMoisture.WET: 87,
        },
        LandCoverType.RURAL: {
            AntecedentMoisture.DRY: 46,
            AntecedentMoisture.NORMAL: 65,
            AntecedentMoisture.WET: 78,
        },
        LandCoverType.FOREST: {
            AntecedentMoisture.DRY: 25,
            AntecedentMoisture.NORMAL: 55,
            AntecedentMoisture.WET: 70,
        },
        LandCoverType.WATER: {
            AntecedentMoisture.DRY: 95,
            AntecedentMoisture.NORMAL: 98,
            AntecedentMoisture.WET: 99,
        },
        LandCoverType.BARREN: {
            AntecedentMoisture.DRY: 72,
            AntecedentMoisture.NORMAL: 86,
            AntecedentMoisture.WET: 91,
        },
    }

    def __init__(self) -> None:
        pass

    def get_cn(self, land_cover: LandCoverType, amc: AntecedentMoisture) -> int:
        """
        Look up the SCS Curve Number for the given land cover and AMC class.

        Parameters
        ----------
        land_cover : LandCoverType
        amc        : AntecedentMoisture

        Returns
        -------
        int
            Curve Number in [1, 100].
        """
        return self.CN_TABLE[land_cover][amc]

    def get_amc_from_antecedent_rainfall(self, rainfall_72h_mm: float) -> AntecedentMoisture:
        """
        Determine the AMC class from 72-hour (≈5-day proxy) antecedent rainfall.

        Thresholds (NRCS NEH Part 630, Ch. 10):
            < 35 mm  → DRY   (AMC I)
            35–53 mm → NORMAL (AMC II)
            > 53 mm  → WET   (AMC III)

        Parameters
        ----------
        rainfall_72h_mm : float
            Total rainfall depth over the prior 72 hours.

        Returns
        -------
        AntecedentMoisture
        """
        if rainfall_72h_mm < 35.0:
            return AntecedentMoisture.DRY
        elif rainfall_72h_mm <= 53.0:
            return AntecedentMoisture.NORMAL
        else:
            return AntecedentMoisture.WET

    def compute_runoff(
        self,
        rainfall_mm: float,
        land_cover: LandCoverType,
        amc: AntecedentMoisture,
    ) -> RunoffEstimate:
        """
        Compute direct runoff depth using the SCS-CN equation.

        Equations (USDA-NRCS NEH Part 630, Ch. 10):
            S  = (25400 / CN) - 254          [retention parameter, mm]
            Ia = 0.2 * S                      [initial abstraction, mm]
            Q  = 0              if P ≤ Ia
            Q  = (P - Ia)² / (P - Ia + S)   if P > Ia

        Parameters
        ----------
        rainfall_mm : float
            Total storm rainfall depth in mm.
        land_cover  : LandCoverType
        amc         : AntecedentMoisture

        Returns
        -------
        RunoffEstimate
        """
        cn = self.get_cn(land_cover, amc)
        cn_safe = max(cn, 1)   # guard against CN = 0

        S = (25400.0 / cn_safe) - 254.0        # retention parameter (mm)
        Ia = 0.2 * S                            # initial abstraction (mm)

        if rainfall_mm <= Ia:
            runoff_mm = 0.0
        else:
            numerator = (rainfall_mm - Ia) ** 2
            denominator = rainfall_mm - Ia + S
            runoff_mm = numerator / denominator

        runoff_mm = max(runoff_mm, 0.0)
        retention_mm = max(rainfall_mm - runoff_mm, 0.0)
        runoff_coefficient = runoff_mm / rainfall_mm if rainfall_mm > 0 else 0.0

        limitations = (
            "CN values are tabulated approximations. "
            "Local calibration against observed hydrograph data improves accuracy. "
            "Method is designed for single-event storm analysis, not continuous simulation."
        )

        return RunoffEstimate(
            rainfall_mm=rainfall_mm,
            runoff_mm=runoff_mm,
            retention_mm=retention_mm,
            runoff_coefficient=runoff_coefficient,
            cn_used=cn,
            land_cover=land_cover.value,
            amc=amc.name,
            limitations=limitations,
        )

    def compute_peak_flow_m3s(
        self,
        runoff_mm: float,
        area_km2: float,
        time_of_concentration_hr: float,
    ) -> float:
        """
        Estimate peak discharge using the SCS triangular unit hydrograph method.

        Formula:
            Qp = 0.208 * A * q / tp

        Where:
            A   = catchment area (km²)
            q   = runoff depth (m)
            tp  = time to peak ≈ time_of_concentration_hr (hr)
            0.208 is the SCS peak rate factor (dimensionless, converted for SI units)

        The result is in m³/s.

        Parameters
        ----------
        runoff_mm              : float – runoff depth in mm
        area_km2               : float – catchment area in km²
        time_of_concentration_hr : float – time of concentration in hours

        Returns
        -------
        float
            Peak flow rate in m³/s.
        """
        if time_of_concentration_hr <= 0 or area_km2 <= 0:
            return 0.0

        runoff_m = runoff_mm / 1000.0
        # SCS formula: Qp (m³/s) = 0.208 * A (km²) * runoff (m) / tp (hr)
        peak_flow = 0.208 * area_km2 * runoff_m / time_of_concentration_hr
        return max(peak_flow, 0.0)

    def estimate_inundation_depth_m(
        self,
        runoff_mm: float,
        area_km2: float,
        depression_fraction: float = 0.1,
    ) -> float:
        """
        Very rough estimate of average inundation depth in metres.

        Assumes that all runoff volume accumulates uniformly over the
        fraction of the catchment that forms topographic depressions.

        Formula:
            volume_m3  = (runoff_mm / 1000) * area_km2 * 1e6
            inund_area = depression_fraction * area_km2 * 1e6  (m²)
            depth      = volume_m3 / inund_area

        IMPORTANT: This is a gross simplification. Actual inundation
        depends on flow dynamics, drainage infrastructure, and local
        topography. Use HEC-RAS or similar for engineering-grade estimates.

        Parameters
        ----------
        runoff_mm          : float – direct runoff depth in mm
        area_km2           : float – catchment area in km²
        depression_fraction: float – fraction [0, 1] of area that stores water

        Returns
        -------
        float
            Estimated average inundation depth in metres.
        """
        if area_km2 <= 0 or depression_fraction <= 0:
            return 0.0

        volume_m3 = (runoff_mm / 1000.0) * area_km2 * 1e6
        inundation_area_m2 = depression_fraction * area_km2 * 1e6
        depth_m = volume_m3 / inundation_area_m2
        return max(depth_m, 0.0)
