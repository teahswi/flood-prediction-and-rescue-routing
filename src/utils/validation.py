"""
src/utils/validation.py — Input-validation helpers for the flood-rescue system.

All public functions return a tuple whose first element is a ``bool``
indicating success, and whose second element carries either an empty string
(on success) or a human-readable error message / list of messages (on failure).
"""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Coordinate validation
# ---------------------------------------------------------------------------

def validate_coordinates(lat: float, lon: float) -> tuple[bool, str]:
    """
    Validate that *lat* and *lon* are within the legal WGS-84 ranges.

    Parameters
    ----------
    lat:
        Latitude in decimal degrees.  Must satisfy ``-90 ≤ lat ≤ 90``.
    lon:
        Longitude in decimal degrees.  Must satisfy ``-180 ≤ lon ≤ 180``.

    Returns
    -------
    tuple[bool, str]
        ``(True, "")`` on success, or ``(False, <error message>)`` on failure.
    """
    if not isinstance(lat, (int, float)):
        return False, f"Latitude must be a number, got {type(lat).__name__!r}."
    if not isinstance(lon, (int, float)):
        return False, f"Longitude must be a number, got {type(lon).__name__!r}."

    errors: list[str] = []
    if not (-90.0 <= lat <= 90.0):
        errors.append(f"Latitude {lat!r} is out of range [-90, 90].")
    if not (-180.0 <= lon <= 180.0):
        errors.append(f"Longitude {lon!r} is out of range [-180, 180].")

    if errors:
        return False, " ".join(errors)
    return True, ""


# ---------------------------------------------------------------------------
# Rainfall validation
# ---------------------------------------------------------------------------

def validate_rainfall_mm(value: float) -> tuple[bool, str]:
    """
    Validate that *value* is a plausible rainfall intensity in mm/hr.

    The upper bound of 500 mm/hr exceeds any recorded rainfall rate and
    is used as a sanity cap to catch data-entry errors.

    Parameters
    ----------
    value:
        Rainfall rate in millimetres per hour.  Must satisfy ``0 ≤ value ≤ 500``.

    Returns
    -------
    tuple[bool, str]
        ``(True, "")`` on success, or ``(False, <error message>)`` on failure.
    """
    if not isinstance(value, (int, float)):
        return False, f"Rainfall must be a number, got {type(value).__name__!r}."
    if value < 0:
        return False, f"Rainfall cannot be negative (got {value!r} mm/hr)."
    if value > 500:
        return False, (
            f"Rainfall {value!r} mm/hr exceeds the plausible maximum of 500 mm/hr. "
            "Check your input."
        )
    return True, ""


# ---------------------------------------------------------------------------
# Radius validation
# ---------------------------------------------------------------------------

def validate_radius_km(radius: float) -> tuple[bool, str]:
    """
    Validate that *radius* is within the accepted study-area radius range.

    Parameters
    ----------
    radius:
        Radius in kilometres.  Must satisfy ``0.5 ≤ radius ≤ 50``.

    Returns
    -------
    tuple[bool, str]
        ``(True, "")`` on success, or ``(False, <error message>)`` on failure.
    """
    if not isinstance(radius, (int, float)):
        return False, f"Radius must be a number, got {type(radius).__name__!r}."
    if radius < 0.5:
        return False, (
            f"Radius {radius!r} km is too small; minimum is 0.5 km."
        )
    if radius > 50.0:
        return False, (
            f"Radius {radius!r} km exceeds the maximum of 50 km. "
            "Large areas may cause memory and performance issues."
        )
    return True, ""


# ---------------------------------------------------------------------------
# Scenario parameters validation
# ---------------------------------------------------------------------------

def validate_scenario_params(params: dict) -> tuple[bool, list[str]]:
    """
    Validate a complete scenario parameter dictionary.

    Expected keys
    -------------
    lat : float
        Latitude of the scenario centre (decimal degrees).
    lon : float
        Longitude of the scenario centre (decimal degrees).
    radius_km : float
        Study-area radius in kilometres.
    rainfall_mm_hr : float
        Peak rainfall intensity in mm/hr.
    duration_hours : float
        Event duration in hours; must satisfy ``0 < duration_hours ≤ 120``.

    Parameters
    ----------
    params:
        Dictionary containing the keys listed above.

    Returns
    -------
    tuple[bool, list[str]]
        ``(True, [])`` if all validations pass, or ``(False, [error, ...])``
        with one entry per failed check.
    """
    errors: list[str] = []

    required_keys = {"lat", "lon", "radius_km", "rainfall_mm_hr", "duration_hours"}
    missing = required_keys - set(params.keys())
    if missing:
        errors.append(f"Missing required keys: {', '.join(sorted(missing))}.")
        # Cannot continue validating missing fields
        return False, errors

    # Coordinates
    ok, msg = validate_coordinates(params["lat"], params["lon"])
    if not ok:
        errors.append(msg)

    # Radius
    ok, msg = validate_radius_km(params["radius_km"])
    if not ok:
        errors.append(msg)

    # Rainfall
    ok, msg = validate_rainfall_mm(params["rainfall_mm_hr"])
    if not ok:
        errors.append(msg)

    # Duration
    duration = params["duration_hours"]
    if not isinstance(duration, (int, float)):
        errors.append(
            f"duration_hours must be a number, got {type(duration).__name__!r}."
        )
    elif duration <= 0:
        errors.append(f"duration_hours must be positive (got {duration!r}).")
    elif duration > 120:
        errors.append(
            f"duration_hours {duration!r} exceeds maximum of 120 hours."
        )

    return (len(errors) == 0), errors


# ---------------------------------------------------------------------------
# String sanitisation
# ---------------------------------------------------------------------------

# Characters considered dangerous in location name strings (shell injection,
# path traversal, HTML injection, etc.)
_DANGEROUS_PATTERN = re.compile(r"[<>\"\'`;|&$\\/()\[\]{}]")


def sanitize_location_name(name: str) -> str:
    """
    Strip potentially dangerous characters from a free-text location name
    and enforce a maximum length.

    Parameters
    ----------
    name:
        Raw location name string from user input.

    Returns
    -------
    str
        Sanitised name, at most 100 characters long.
    """
    if not isinstance(name, str):
        name = str(name)

    # Remove dangerous characters
    sanitized = _DANGEROUS_PATTERN.sub("", name)

    # Collapse repeated whitespace
    sanitized = re.sub(r"\s+", " ", sanitized).strip()

    # Enforce length limit
    return sanitized[:100]


# ---------------------------------------------------------------------------
# Incident report validation
# ---------------------------------------------------------------------------

def validate_incident_report(report: dict) -> tuple[bool, list[str]]:
    """
    Validate a structured incident report submitted by a field operator.

    Required structure
    ------------------
    .. code-block:: python

        {
            "location": {"lat": float, "lon": float},
            "severity": int,        # 1 (minor) – 5 (catastrophic)
            "description": str,     # free-text, 1–500 characters
        }

    Optional keys are ignored.

    Parameters
    ----------
    report:
        Dictionary containing the keys described above.

    Returns
    -------
    tuple[bool, list[str]]
        ``(True, [])`` if the report is valid, otherwise ``(False, [error, ...])``.
    """
    errors: list[str] = []

    if not isinstance(report, dict):
        return False, [f"Report must be a dict, got {type(report).__name__!r}."]

    # --- location ---
    location = report.get("location")
    if location is None:
        errors.append("Missing required key 'location'.")
    elif not isinstance(location, dict):
        errors.append(
            f"'location' must be a dict with 'lat' and 'lon' keys, "
            f"got {type(location).__name__!r}."
        )
    else:
        lat = location.get("lat")
        lon = location.get("lon")
        if lat is None:
            errors.append("'location' is missing key 'lat'.")
        if lon is None:
            errors.append("'location' is missing key 'lon'.")
        if lat is not None and lon is not None:
            ok, msg = validate_coordinates(lat, lon)
            if not ok:
                errors.append(f"Invalid location coordinates: {msg}")

    # --- severity ---
    severity = report.get("severity")
    if severity is None:
        errors.append("Missing required key 'severity'.")
    elif not isinstance(severity, int):
        errors.append(
            f"'severity' must be an integer (1–5), got {type(severity).__name__!r}."
        )
    elif severity < 1 or severity > 5:
        errors.append(
            f"'severity' must be in the range 1–5 (got {severity!r})."
        )

    # --- description ---
    description = report.get("description")
    if description is None:
        errors.append("Missing required key 'description'.")
    elif not isinstance(description, str):
        errors.append(
            f"'description' must be a string, got {type(description).__name__!r}."
        )
    elif len(description.strip()) == 0:
        errors.append("'description' must not be empty.")
    elif len(description) > 500:
        errors.append(
            f"'description' is too long ({len(description)} chars); maximum is 500."
        )

    return (len(errors) == 0), errors
