"""
Emergency incident management for flood rescue operations.

Incident priority formula:
    P = a*S + b*F + c*V + d*A

Where:
    S: severity of reported emergency [1-5]
    F: flood threat level at incident location [0-1]
    V: vulnerability factor (elderly/children/medical/etc.) [0-1]
    A: urgency based on time remaining vs time to safety [0-1]

NOTE: This priority score is a decision-SUPPORT tool, not a validated
triage protocol. All priorities must be confirmed by trained emergency
responders. The system does not replace professional emergency management.
"""
import uuid
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum


class IncidentStatus(Enum):
    REPORTED = "reported"
    ACKNOWLEDGED = "acknowledged"
    ASSIGNED = "assigned"
    EN_ROUTE = "en_route"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


class IncidentSeverity(Enum):
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4
    LIFE_THREATENING = 5


@dataclass
class Incident:
    incident_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    lat: float = 0.0
    lon: float = 0.0
    severity: int = 2  # 1-5
    description: str = ""
    people_count: int = 1
    has_elderly: bool = False
    has_children: bool = False
    has_medical_needs: bool = False
    reported_at: datetime = field(default_factory=datetime.utcnow)
    status: IncidentStatus = IncidentStatus.REPORTED
    flood_risk_at_location: float = 0.0
    assigned_team_id: Optional[str] = None
    assigned_shelter_id: Optional[str] = None
    estimated_arrival_minutes: Optional[float] = None
    time_to_flooding_minutes: Optional[float] = None  # from simulation
    priority_score: float = 0.0
    audit_log: List[str] = field(default_factory=list)

    def compute_priority(self, weights: dict = None) -> float:
        """
        Compute priority score using a weighted formula.

        Default weights: severity=0.35, flood=0.30, vulnerability=0.20, urgency=0.15
        Returns a float in [0, 1].

        NOTE: This is a decision-support tool only. Do not use as sole basis for
        triage decisions without professional emergency management review.
        """
        if weights is None:
            weights = {
                "severity": 0.35,
                "flood": 0.30,
                "vulnerability": 0.20,
                "urgency": 0.15,
            }

        a = weights.get("severity", 0.35)
        b = weights.get("flood", 0.30)
        c = weights.get("vulnerability", 0.20)
        d = weights.get("urgency", 0.15)

        # S: normalise severity to [0, 1]
        S = self.severity / 5.0

        # F: flood risk at location (already in [0, 1])
        F = float(self.flood_risk_at_location)

        # V: vulnerability composite
        V = (
            (0.3 if self.has_elderly else 0.0)
            + (0.3 if self.has_children else 0.0)
            + (0.4 if self.has_medical_needs else 0.0)
        )
        # Add a bonus for group size, capped at 0.3
        V = min(V + min(self.people_count / 10.0, 0.3), 1.0)

        # A: urgency based on time to flooding
        if self.time_to_flooding_minutes is None:
            A = 0.5  # unknown
        elif self.time_to_flooding_minutes < 30:
            A = 1.0  # imminent
        elif self.time_to_flooding_minutes < 60:
            A = 0.75
        elif self.time_to_flooding_minutes < 120:
            A = 0.5
        else:
            A = 0.25

        P = a * S + b * F + c * V + d * A
        P = float(np.clip(P, 0.0, 1.0))
        self.priority_score = P
        return P

    def add_audit_entry(self, action: str, actor: str = "system") -> None:
        """Append a timestamped audit entry."""
        entry = f"[{datetime.utcnow().isoformat()}] {actor}: {action}"
        self.audit_log.append(entry)

    def update_status(self, new_status: IncidentStatus, actor: str = "system") -> None:
        """Update the incident status and record in the audit log."""
        self.status = new_status
        self.add_audit_entry(f"Status changed to {new_status.value}", actor=actor)

    def time_pressure_warning(self) -> Optional[str]:
        """
        Return a warning string if the estimated arrival time exceeds the
        predicted time to flooding; otherwise return None.
        """
        if (
            self.time_to_flooding_minutes is not None
            and self.estimated_arrival_minutes is not None
        ):
            if self.estimated_arrival_minutes > self.time_to_flooding_minutes:
                return (
                    f"WARNING: Estimated arrival ({self.estimated_arrival_minutes:.0f} min) "
                    f"exceeds predicted time to flooding ({self.time_to_flooding_minutes:.0f} min). "
                    f"URGENT REVIEW NEEDED."
                )
        return None


class IncidentManager:
    """
    Central registry and manager for emergency incidents.
    """

    def __init__(self):
        self.incidents: Dict[str, Incident] = {}

    def add_incident(self, incident: Incident) -> str:
        """Add an incident to the registry. Returns the incident ID."""
        self.incidents[incident.incident_id] = incident
        incident.add_audit_entry("Incident created and registered")
        return incident.incident_id

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Retrieve an incident by ID, or None if not found."""
        return self.incidents.get(incident_id)

    def update_incident_risk(self, incident_id: str, flood_risk: float) -> None:
        """Update the flood risk value at a specific incident location."""
        incident = self.incidents.get(incident_id)
        if incident is not None:
            incident.flood_risk_at_location = float(np.clip(flood_risk, 0.0, 1.0))
            incident.add_audit_entry(
                f"Flood risk updated to {incident.flood_risk_at_location:.3f}"
            )

    def assign_team(
        self,
        incident_id: str,
        team_id: str,
        shelter_id: str,
        eta_minutes: float,
        actor: str = "dispatcher",
    ) -> None:
        """
        Assign a rescue team and destination shelter to an incident and update
        the estimated arrival time.
        """
        incident = self.incidents.get(incident_id)
        if incident is None:
            raise KeyError(f"Incident '{incident_id}' not found.")
        incident.assigned_team_id = team_id
        incident.assigned_shelter_id = shelter_id
        incident.estimated_arrival_minutes = float(eta_minutes)
        incident.update_status(IncidentStatus.ASSIGNED, actor=actor)
        incident.add_audit_entry(
            f"Assigned to team '{team_id}', shelter '{shelter_id}', ETA {eta_minutes:.1f} min",
            actor=actor,
        )

    def get_sorted_by_priority(self) -> List[Incident]:
        """
        Recompute priority scores for all incidents and return them sorted
        from highest to lowest priority.
        """
        incidents = list(self.incidents.values())
        for inc in incidents:
            inc.compute_priority()
        return sorted(incidents, key=lambda i: i.priority_score, reverse=True)

    def get_active_incidents(self) -> List[Incident]:
        """Return incidents that are not yet resolved or cancelled."""
        inactive = {IncidentStatus.RESOLVED, IncidentStatus.CANCELLED}
        return [i for i in self.incidents.values() if i.status not in inactive]

    def get_incidents_dataframe(self) -> pd.DataFrame:
        """Return a tabular summary of all incidents as a DataFrame."""
        rows = []
        for inc in self.incidents.values():
            rows.append(
                {
                    "incident_id": inc.incident_id,
                    "lat": inc.lat,
                    "lon": inc.lon,
                    "severity": inc.severity,
                    "people_count": inc.people_count,
                    "has_elderly": inc.has_elderly,
                    "has_children": inc.has_children,
                    "has_medical_needs": inc.has_medical_needs,
                    "status": inc.status.value,
                    "flood_risk_at_location": inc.flood_risk_at_location,
                    "priority_score": inc.priority_score,
                    "assigned_team_id": inc.assigned_team_id,
                    "assigned_shelter_id": inc.assigned_shelter_id,
                    "estimated_arrival_minutes": inc.estimated_arrival_minutes,
                    "time_to_flooding_minutes": inc.time_to_flooding_minutes,
                    "reported_at": inc.reported_at.isoformat(),
                }
            )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def get_urgent_mismatches(self) -> List[Incident]:
        """
        Return incidents where the estimated arrival time exceeds the predicted
        time to flooding — i.e., the team may arrive too late.
        """
        mismatches = []
        for inc in self.incidents.values():
            if (
                inc.time_to_flooding_minutes is not None
                and inc.estimated_arrival_minutes is not None
                and inc.estimated_arrival_minutes > inc.time_to_flooding_minutes
            ):
                mismatches.append(inc)
        return mismatches


def create_sample_incidents(
    center_lat: float, center_lon: float, count: int = 5
) -> List[Incident]:
    """
    Generate a set of realistic sample incidents near a geographic centre.

    Parameters
    ----------
    center_lat : float
        Latitude of the area centre.
    center_lon : float
        Longitude of the area centre.
    count : int
        Number of incidents to generate.

    Returns
    -------
    List[Incident]
    """
    rng = np.random.default_rng(42)

    templates = [
        {
            "description": "Family trapped on roof of flooded house",
            "severity": 5,
            "people_count": 4,
            "has_elderly": True,
            "has_children": True,
            "has_medical_needs": False,
            "flood_risk_at_location": 0.9,
            "time_to_flooding_minutes": 20.0,
        },
        {
            "description": "Elderly resident unable to evacuate",
            "severity": 4,
            "people_count": 2,
            "has_elderly": True,
            "has_children": False,
            "has_medical_needs": True,
            "flood_risk_at_location": 0.75,
            "time_to_flooding_minutes": 45.0,
        },
        {
            "description": "Group stranded at community centre",
            "severity": 3,
            "people_count": 15,
            "has_elderly": False,
            "has_children": True,
            "has_medical_needs": False,
            "flood_risk_at_location": 0.55,
            "time_to_flooding_minutes": 90.0,
        },
        {
            "description": "Vehicle stalled in rising water",
            "severity": 3,
            "people_count": 2,
            "has_elderly": False,
            "has_children": False,
            "has_medical_needs": False,
            "flood_risk_at_location": 0.8,
            "time_to_flooding_minutes": 15.0,
        },
        {
            "description": "Person with medical equipment requiring power",
            "severity": 4,
            "people_count": 1,
            "has_elderly": False,
            "has_children": False,
            "has_medical_needs": True,
            "flood_risk_at_location": 0.4,
            "time_to_flooding_minutes": 180.0,
        },
        {
            "description": "Collapsed section blocking evacuation route",
            "severity": 2,
            "people_count": 6,
            "has_elderly": False,
            "has_children": False,
            "has_medical_needs": False,
            "flood_risk_at_location": 0.3,
            "time_to_flooding_minutes": None,
        },
        {
            "description": "Infant and parent isolated by flood water",
            "severity": 5,
            "people_count": 2,
            "has_elderly": False,
            "has_children": True,
            "has_medical_needs": False,
            "flood_risk_at_location": 0.85,
            "time_to_flooding_minutes": 25.0,
        },
        {
            "description": "Resident refusing to evacuate, needs persuasion",
            "severity": 1,
            "people_count": 1,
            "has_elderly": True,
            "has_children": False,
            "has_medical_needs": False,
            "flood_risk_at_location": 0.5,
            "time_to_flooding_minutes": 120.0,
        },
    ]

    incidents: List[Incident] = []
    for i in range(count):
        tmpl = templates[i % len(templates)]
        lat_offset = rng.uniform(-0.05, 0.05)
        lon_offset = rng.uniform(-0.05, 0.05)
        inc = Incident(
            lat=center_lat + lat_offset,
            lon=center_lon + lon_offset,
            severity=tmpl["severity"],
            description=tmpl["description"],
            people_count=tmpl["people_count"],
            has_elderly=tmpl["has_elderly"],
            has_children=tmpl["has_children"],
            has_medical_needs=tmpl["has_medical_needs"],
            flood_risk_at_location=tmpl["flood_risk_at_location"],
            time_to_flooding_minutes=tmpl["time_to_flooding_minutes"],
        )
        inc.compute_priority()
        incidents.append(inc)

    return incidents
