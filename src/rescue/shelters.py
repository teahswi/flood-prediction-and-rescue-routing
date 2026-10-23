"""
Emergency shelter management for flood rescue operations.
"""
import uuid
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable
from datetime import datetime
from enum import Enum
import math


class ShelterStatus(Enum):
    AVAILABLE = "available"
    FULL = "full"
    HAZARDOUS = "hazardous"   # shelter itself is at flood risk
    CLOSED = "closed"


@dataclass
class Shelter:
    shelter_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    lat: float = 0.0
    lon: float = 0.0
    capacity: int = 100
    current_occupancy: int = 0
    status: ShelterStatus = ShelterStatus.AVAILABLE
    shelter_type: str = "community_center"  # school, hospital, stadium, etc.
    flood_risk: float = 0.0
    has_medical: bool = False
    has_food: bool = True
    has_water: bool = True
    accessibility: str = "full"  # "full", "limited", "none"
    contact_phone: str = ""
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def available_spaces(self) -> int:
        """Return the number of remaining spaces at this shelter."""
        return max(0, self.capacity - self.current_occupancy)

    def occupancy_pct(self) -> float:
        """Return current occupancy as a percentage of capacity."""
        if self.capacity == 0:
            return 100.0
        return self.current_occupancy / self.capacity * 100.0

    def is_accepting(self) -> bool:
        """
        Return True when the shelter can receive new evacuees — i.e. it is
        available, has free spaces, and is not at high flood risk.
        """
        return (
            self.status == ShelterStatus.AVAILABLE
            and self.available_spaces() > 0
            and self.flood_risk < 0.7
        )

    def update_status_from_risk(self) -> None:
        """
        Automatically set status to HAZARDOUS when the shelter's own flood
        risk exceeds 0.7.
        """
        if self.flood_risk > 0.7:
            self.status = ShelterStatus.HAZARDOUS
        self.last_updated = datetime.utcnow()

    def to_dict(self) -> dict:
        """Return a plain-dict representation of the shelter."""
        return {
            "shelter_id": self.shelter_id,
            "name": self.name,
            "lat": self.lat,
            "lon": self.lon,
            "capacity": self.capacity,
            "current_occupancy": self.current_occupancy,
            "available_spaces": self.available_spaces(),
            "occupancy_pct": round(self.occupancy_pct(), 1),
            "status": self.status.value,
            "shelter_type": self.shelter_type,
            "flood_risk": self.flood_risk,
            "has_medical": self.has_medical,
            "has_food": self.has_food,
            "has_water": self.has_water,
            "accessibility": self.accessibility,
            "contact_phone": self.contact_phone,
            "is_accepting": self.is_accepting(),
            "last_updated": self.last_updated.isoformat(),
        }


class ShelterRegistry:
    """
    Central registry for all emergency shelters in the area of operations.
    """

    def __init__(self):
        self.shelters: Dict[str, Shelter] = {}

    def add_shelter(self, shelter: Shelter) -> str:
        """Add a shelter to the registry. Returns the shelter ID."""
        self.shelters[shelter.shelter_id] = shelter
        return shelter.shelter_id

    def get_shelter(self, shelter_id: str) -> Optional[Shelter]:
        """Retrieve a shelter by ID, or None if not found."""
        return self.shelters.get(shelter_id)

    def update_occupancy(self, shelter_id: str, new_occupancy: int) -> None:
        """
        Update the current occupancy for a shelter, clamping to [0, capacity],
        and refresh status if the shelter is now full.
        """
        shelter = self.shelters.get(shelter_id)
        if shelter is None:
            raise KeyError(f"Shelter '{shelter_id}' not found.")
        shelter.current_occupancy = max(0, min(new_occupancy, shelter.capacity))
        if shelter.current_occupancy >= shelter.capacity:
            if shelter.status == ShelterStatus.AVAILABLE:
                shelter.status = ShelterStatus.FULL
        elif shelter.status == ShelterStatus.FULL:
            shelter.status = ShelterStatus.AVAILABLE
        shelter.last_updated = datetime.utcnow()

    def update_shelter_risks(self, risk_fn: Callable[[float, float], float]) -> None:
        """
        Update the flood_risk attribute for each shelter by calling ``risk_fn``
        with the shelter's (lat, lon).  Also triggers status update.

        Parameters
        ----------
        risk_fn : Callable[[float, float], float]
            A function that accepts (lat, lon) and returns a risk value in [0,1].
        """
        for shelter in self.shelters.values():
            shelter.flood_risk = float(np.clip(risk_fn(shelter.lat, shelter.lon), 0.0, 1.0))
            shelter.update_status_from_risk()

    def get_available_shelters(self) -> List[Shelter]:
        """Return a list of shelters that are currently accepting evacuees."""
        return [s for s in self.shelters.values() if s.is_accepting()]

    def get_nearest_available(
        self, lat: float, lon: float, max_results: int = 5
    ) -> List[Shelter]:
        """
        Return up to *max_results* available shelters sorted by Haversine
        distance from the given (lat, lon).
        """
        available = self.get_available_shelters()
        ranked = sorted(
            available,
            key=lambda s: haversine_distance_km(lat, lon, s.lat, s.lon),
        )
        return ranked[:max_results]

    def get_shelters_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with one row per shelter."""
        if not self.shelters:
            return pd.DataFrame()
        return pd.DataFrame([s.to_dict() for s in self.shelters.values()])

    def registry_summary(self) -> dict:
        """
        Return a summary dictionary with aggregate shelter statistics.

        Keys: total, available, full, hazardous, closed,
              total_capacity, total_occupancy
        """
        shelters = list(self.shelters.values())
        return {
            "total": len(shelters),
            "available": sum(1 for s in shelters if s.status == ShelterStatus.AVAILABLE),
            "full": sum(1 for s in shelters if s.status == ShelterStatus.FULL),
            "hazardous": sum(1 for s in shelters if s.status == ShelterStatus.HAZARDOUS),
            "closed": sum(1 for s in shelters if s.status == ShelterStatus.CLOSED),
            "total_capacity": sum(s.capacity for s in shelters),
            "total_occupancy": sum(s.current_occupancy for s in shelters),
        }


def haversine_distance_km(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """
    Compute the great-circle distance between two points on the Earth's
    surface using the Haversine formula.

    Parameters
    ----------
    lat1, lon1 : float
        Latitude and longitude of the first point (degrees).
    lat2, lon2 : float
        Latitude and longitude of the second point (degrees).

    Returns
    -------
    float
        Distance in kilometres.
    """
    R = 6371.0  # Earth's mean radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def create_sample_shelters(
    center_lat: float, center_lon: float, count: int = 8
) -> List[Shelter]:
    """
    Generate a set of realistic sample shelters near a geographic centre.

    Parameters
    ----------
    center_lat : float
        Latitude of the area centre.
    center_lon : float
        Longitude of the area centre.
    count : int
        Number of shelters to generate.

    Returns
    -------
    List[Shelter]
    """
    rng = np.random.default_rng(7)

    templates = [
        {
            "name": "City High School",
            "shelter_type": "school",
            "capacity": 500,
            "current_occupancy": 120,
            "has_medical": False,
            "has_food": True,
            "has_water": True,
            "accessibility": "full",
            "flood_risk": 0.1,
            "contact_phone": "555-0101",
        },
        {
            "name": "Community Sports Hall",
            "shelter_type": "stadium",
            "capacity": 1200,
            "current_occupancy": 400,
            "has_medical": True,
            "has_food": True,
            "has_water": True,
            "accessibility": "full",
            "flood_risk": 0.05,
            "contact_phone": "555-0102",
        },
        {
            "name": "District Hospital Annex",
            "shelter_type": "hospital",
            "capacity": 200,
            "current_occupancy": 180,
            "has_medical": True,
            "has_food": True,
            "has_water": True,
            "accessibility": "full",
            "flood_risk": 0.2,
            "contact_phone": "555-0103",
        },
        {
            "name": "Old Town Community Centre",
            "shelter_type": "community_center",
            "capacity": 300,
            "current_occupancy": 50,
            "has_medical": False,
            "has_food": True,
            "has_water": True,
            "accessibility": "limited",
            "flood_risk": 0.65,
            "contact_phone": "555-0104",
        },
        {
            "name": "Riverside Church Hall",
            "shelter_type": "community_center",
            "capacity": 150,
            "current_occupancy": 0,
            "has_medical": False,
            "has_food": False,
            "has_water": True,
            "accessibility": "full",
            "flood_risk": 0.75,  # at risk — will become HAZARDOUS
            "contact_phone": "555-0105",
        },
        {
            "name": "Uptown Convention Centre",
            "shelter_type": "stadium",
            "capacity": 2000,
            "current_occupancy": 350,
            "has_medical": True,
            "has_food": True,
            "has_water": True,
            "accessibility": "full",
            "flood_risk": 0.02,
            "contact_phone": "555-0106",
        },
        {
            "name": "South Primary School",
            "shelter_type": "school",
            "capacity": 400,
            "current_occupancy": 400,  # full
            "has_medical": False,
            "has_food": True,
            "has_water": True,
            "accessibility": "full",
            "flood_risk": 0.15,
            "contact_phone": "555-0107",
        },
        {
            "name": "West Side Fire Station",
            "shelter_type": "emergency_station",
            "capacity": 80,
            "current_occupancy": 10,
            "has_medical": True,
            "has_food": True,
            "has_water": True,
            "accessibility": "full",
            "flood_risk": 0.08,
            "contact_phone": "555-0108",
        },
        {
            "name": "Hilltop Recreation Centre",
            "shelter_type": "community_center",
            "capacity": 600,
            "current_occupancy": 200,
            "has_medical": False,
            "has_food": True,
            "has_water": True,
            "accessibility": "limited",
            "flood_risk": 0.01,
            "contact_phone": "555-0109",
        },
        {
            "name": "Northern Trade Exhibition Hall",
            "shelter_type": "stadium",
            "capacity": 3000,
            "current_occupancy": 900,
            "has_medical": True,
            "has_food": True,
            "has_water": True,
            "accessibility": "full",
            "flood_risk": 0.03,
            "contact_phone": "555-0110",
        },
    ]

    shelters: List[Shelter] = []
    for i in range(count):
        tmpl = templates[i % len(templates)]
        lat_offset = rng.uniform(-0.08, 0.08)
        lon_offset = rng.uniform(-0.08, 0.08)

        shelter = Shelter(
            name=tmpl["name"],
            lat=center_lat + lat_offset,
            lon=center_lon + lon_offset,
            capacity=tmpl["capacity"],
            current_occupancy=tmpl["current_occupancy"],
            shelter_type=tmpl["shelter_type"],
            flood_risk=tmpl["flood_risk"],
            has_medical=tmpl["has_medical"],
            has_food=tmpl["has_food"],
            has_water=tmpl["has_water"],
            accessibility=tmpl["accessibility"],
            contact_phone=tmpl["contact_phone"],
        )

        # Derive initial status
        if shelter.current_occupancy >= shelter.capacity:
            shelter.status = ShelterStatus.FULL
        elif shelter.flood_risk > 0.7:
            shelter.status = ShelterStatus.HAZARDOUS
        else:
            shelter.status = ShelterStatus.AVAILABLE

        shelters.append(shelter)

    return shelters
