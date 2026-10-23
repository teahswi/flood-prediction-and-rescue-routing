"""
Flood Prediction & Rescue Routing System
========================================
A geospatial decision-support system for flood prediction, spread simulation,
and emergency rescue routing.

DISCLAIMER: This is a prototype decision-support tool. All predictions and
routes are estimates based on available data and simplified models. They are
NOT authoritative emergency instructions. All recommendations must be verified
by qualified emergency management professionals.

Data sources: Open-Meteo (weather), Open-Topo-Data (terrain/SRTM), OpenStreetMap (roads)
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Page config MUST be first streamlit call
st.set_page_config(
    page_title="Flood Prediction & Rescue Routing",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CSS STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Dark sidebar */
[data-testid="stSidebar"] {
    background-color: #1a1a2e;
    color: #ffffff;
}
[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}
[data-testid="stSidebar"] .stButton > button {
    background-color: #16213e;
    color: #ffffff !important;
    border: 1px solid #0f3460;
    border-radius: 6px;
    width: 100%;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #0f3460;
    border-color: #e94560;
}
[data-testid="stSidebar"] label {
    color: #b0b8c8 !important;
    font-size: 0.85rem !important;
}
[data-testid="stSidebar"] .stSlider > div > div > div {
    background-color: #e94560;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
    color: #ffffff !important;
}
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stNumberInput input {
    background-color: #16213e !important;
    color: #ffffff !important;
    border: 1px solid #0f3460 !important;
}

/* Status pills */
.pill-safe    { background:#2ecc71; color:#fff; padding:2px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
.pill-moderate{ background:#f39c12; color:#fff; padding:2px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
.pill-high    { background:#e74c3c; color:#fff; padding:2px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
.pill-critical{ background:#8e44ad; color:#fff; padding:2px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #16213e 100%);
    border-left: 4px solid #e94560;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 6px 0;
    color: #ffffff;
}
.metric-card .metric-label { font-size: 0.75rem; color: #b0b8c8; text-transform: uppercase; letter-spacing: 0.8px; }
.metric-card .metric-value { font-size: 1.6rem; font-weight: 700; color: #ffffff; margin-top: 2px; }
.metric-card .metric-delta { font-size: 0.75rem; margin-top: 2px; }
.metric-green  { border-left-color: #2ecc71; }
.metric-yellow { border-left-color: #f39c12; }
.metric-red    { border-left-color: #e74c3c; }
.metric-purple { border-left-color: #8e44ad; }

/* Warning banner */
.warning-banner {
    background-color: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 6px;
    padding: 10px 16px;
    margin: 8px 0;
    color: #856404;
    font-size: 0.88rem;
}
.critical-banner {
    background-color: #f8d7da;
    border: 1px solid #f5c2c7;
    border-radius: 6px;
    padding: 10px 16px;
    margin: 8px 0;
    color: #842029;
    font-size: 0.88rem;
}
.info-banner {
    background-color: #cff4fc;
    border: 1px solid #9eeaf9;
    border-radius: 6px;
    padding: 10px 16px;
    margin: 8px 0;
    color: #055160;
    font-size: 0.88rem;
}

/* Live indicator */
.live-indicator {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #2ecc71;
}
.live-dot {
    width: 9px;
    height: 9px;
    background: #2ecc71;
    border-radius: 50%;
    display: inline-block;
    animation: pulse-green 1.6s infinite;
}
@keyframes pulse-green {
    0%   { box-shadow: 0 0 0 0 rgba(46,204,113,.7); }
    70%  { box-shadow: 0 0 0 7px rgba(46,204,113,0); }
    100% { box-shadow: 0 0 0 0 rgba(46,204,113,0); }
}

/* Incident cards */
.incident-card {
    background: #fff;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 12px 14px;
    margin: 6px 0;
    border-left: 5px solid #e74c3c;
}
.shelter-card {
    background: #fff;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 12px 14px;
    margin: 6px 0;
    border-left: 5px solid #2ecc71;
}

/* Footer */
.footer {
    background: #1a1a2e;
    color: #8899aa;
    padding: 16px 24px;
    border-radius: 8px;
    font-size: 0.78rem;
    margin-top: 24px;
    text-align: center;
}
.footer a { color: #5dade2; text-decoration: none; }

/* Tab labels */
button[data-baseweb="tab"] {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}

/* Progress bar colour override */
.stProgress > div > div > div > div {
    background-color: #2ecc71;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
def _sample_incidents():
    return [
        {
            "id": "INC-001",
            "lat": 19.0550, "lon": 72.8400,
            "severity": 4, "people_count": 12,
            "elderly": True, "children": True, "medical": False,
            "description": "Families stranded on rooftop, 1.2m water on ground floor",
            "status": "Active",
            "reported_at": (datetime.utcnow() - timedelta(minutes=38)).isoformat(),
            "priority_score": 0.87,
        },
        {
            "id": "INC-002",
            "lat": 19.0820, "lon": 72.8850,
            "severity": 3, "people_count": 5,
            "elderly": False, "children": False, "medical": True,
            "description": "Road blocked by debris, medical emergency – diabetic patient",
            "status": "Active",
            "reported_at": (datetime.utcnow() - timedelta(minutes=15)).isoformat(),
            "priority_score": 0.74,
        },
        {
            "id": "INC-003",
            "lat": 19.0200, "lon": 72.8500,
            "severity": 2, "people_count": 8,
            "elderly": False, "children": True, "medical": False,
            "description": "Vehicle stalled mid-road in flooded underpass",
            "status": "In Progress",
            "reported_at": (datetime.utcnow() - timedelta(minutes=62)).isoformat(),
            "priority_score": 0.56,
        },
        {
            "id": "INC-004",
            "lat": 19.1050, "lon": 72.8620,
            "severity": 5, "people_count": 30,
            "elderly": True, "children": True, "medical": True,
            "description": "Old-age home partially submerged, urgent evacuation needed",
            "status": "Active",
            "reported_at": (datetime.utcnow() - timedelta(minutes=8)).isoformat(),
            "priority_score": 0.97,
        },
        {
            "id": "INC-005",
            "lat": 18.9900, "lon": 72.8350,
            "severity": 1, "people_count": 2,
            "elderly": False, "children": False, "medical": False,
            "description": "Minor flooding on street, residents request monitoring",
            "status": "Resolved",
            "reported_at": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
            "priority_score": 0.15,
        },
    ]


def _sample_shelters():
    return [
        {
            "id": "SHE-001",
            "name": "Bandra Community Hall",
            "type": "Government Hall",
            "lat": 19.0596, "lon": 72.8295,
            "capacity": 500, "current_occupancy": 312,
            "status": "Open",
            "flood_risk_score": 0.18,
            "facilities": ["food", "water", "medical", "power"],
            "contact": "+91-22-26401234",
        },
        {
            "id": "SHE-002",
            "name": "Andheri Sports Complex",
            "type": "Sports Facility",
            "lat": 19.1136, "lon": 72.8697,
            "capacity": 800, "current_occupancy": 621,
            "status": "Near Capacity",
            "flood_risk_score": 0.23,
            "facilities": ["food", "water", "power"],
            "contact": "+91-22-26834567",
        },
        {
            "id": "SHE-003",
            "name": "Chembur School Ground",
            "type": "School Ground",
            "lat": 19.0627, "lon": 72.8997,
            "capacity": 300, "current_occupancy": 88,
            "status": "Open",
            "flood_risk_score": 0.11,
            "facilities": ["food", "water", "medical", "power", "toilets"],
            "contact": "+91-22-25204567",
        },
        {
            "id": "SHE-004",
            "name": "Kurla Railway Ground",
            "type": "Railway Ground",
            "lat": 19.0726, "lon": 72.8793,
            "capacity": 600, "current_occupancy": 597,
            "status": "Full",
            "flood_risk_score": 0.38,
            "facilities": ["water"],
            "contact": "+91-22-25016789",
        },
        {
            "id": "SHE-005",
            "name": "Borivali National Park Entry",
            "type": "Open Ground",
            "lat": 19.2288, "lon": 72.8592,
            "capacity": 1200, "current_occupancy": 204,
            "status": "Open",
            "flood_risk_score": 0.07,
            "facilities": ["water", "toilets"],
            "contact": "+91-22-28974567",
        },
    ]


def _init_session_state():
    defaults = {
        "study_area": {
            "name": "Mumbai, Maharashtra",
            "lat": 19.0760,
            "lon": 72.8777,
            "radius_km": 15.0,
        },
        "weather_data": None,
        "terrain_grid": None,
        "road_graph": None,
        "simulation_result": None,
        "route_comparison": None,
        "incidents": _sample_incidents(),
        "shelters": _sample_shelters(),
        "last_refresh": datetime.utcnow(),
        "scenario_rainfall": 25.0,
        "scenario_duration": 6.0,
        "scenario_land_cover": "Suburban",
        "selected_start": None,
        "selected_end": None,
        "show_flood_risk_layer": True,
        "show_shelters": True,
        "show_incidents": True,
        "show_route": True,
        "data_loaded": False,
        "lambda_weight": 2.0,
        "risk_threshold_low": 0.3,
        "risk_threshold_high": 0.6,
        "dispatch_log": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session_state()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo / title
    st.markdown("""
    <div style="text-align:center; padding: 12px 0 20px 0;">
        <div style="font-size:2.8rem;">🌊</div>
        <div style="font-size:1.1rem; font-weight:700; color:#ffffff; letter-spacing:0.5px;">
            FloodGuard AI
        </div>
        <div style="font-size:0.72rem; color:#8899aa; margin-top:2px;">
            Prediction &amp; Rescue Routing
        </div>
    </div>
    <hr style="border-color:#0f3460; margin:0 0 16px 0;">
    """, unsafe_allow_html=True)

    # ── Study Area ──────────────────────────────────────────────────────────
    st.markdown("### 📍 Study Area")
    area_name = st.text_input(
        "City / Location",
        value=st.session_state.study_area["name"],
        key="sb_area_name",
    )
    col_lat, col_lon = st.columns(2)
    with col_lat:
        area_lat = st.number_input(
            "Latitude", value=st.session_state.study_area["lat"],
            min_value=-90.0, max_value=90.0, step=0.001,
            format="%.4f", key="sb_lat",
        )
    with col_lon:
        area_lon = st.number_input(
            "Longitude", value=st.session_state.study_area["lon"],
            min_value=-180.0, max_value=180.0, step=0.001,
            format="%.4f", key="sb_lon",
        )
    area_radius = st.slider(
        "Radius (km)", min_value=1, max_value=50,
        value=int(st.session_state.study_area["radius_km"]),
        key="sb_radius",
    )
    if st.button("📌 Load Area", key="btn_load_area"):
        st.session_state.study_area = {
            "name": area_name,
            "lat": area_lat,
            "lon": area_lon,
            "radius_km": float(area_radius),
        }
        st.session_state.weather_data = None
        st.session_state.data_loaded = False
        st.session_state.last_refresh = datetime.utcnow()
        st.success(f"Area set to {area_name}")

    st.markdown("<hr style='border-color:#0f3460; margin:16px 0;'>", unsafe_allow_html=True)

    # ── Data Status ─────────────────────────────────────────────────────────
    st.markdown("### 📡 Data Status")

    def _freshness_pill(label, fetched_at_iso, error=False):
        if error:
            color, text = "#e74c3c", "Error"
        elif fetched_at_iso is None:
            color, text = "#95a5a6", "No data"
        else:
            age_min = (datetime.utcnow() - datetime.fromisoformat(fetched_at_iso)).total_seconds() / 60
            if age_min < 30:
                color, text = "#2ecc71", f"Fresh ({int(age_min)}m ago)"
            else:
                color, text = "#f39c12", f"Stale ({int(age_min)}m ago)"
        return (
            f'<span style="display:inline-block;background:{color};color:#fff;'
            f'padding:1px 8px;border-radius:10px;font-size:0.72rem;font-weight:600;">'
            f'{text}</span>'
        )

    weather_fetched = (
        st.session_state.weather_data.get("fetched_at")
        if st.session_state.weather_data else None
    )
    terrain_ok = st.session_state.terrain_grid is not None
    road_ok = st.session_state.road_graph is not None

    st.markdown(
        f"🌧️ **Weather** &nbsp; {_freshness_pill('Weather', weather_fetched)}",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"🏔️ **Terrain** &nbsp; "
        f"{_freshness_pill('Terrain', datetime.utcnow().isoformat() if terrain_ok else None)}",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"🛣️ **Road Graph** &nbsp; "
        f"{_freshness_pill('Roads', datetime.utcnow().isoformat() if road_ok else None)}",
        unsafe_allow_html=True,
    )

    refresh_ago = (datetime.utcnow() - st.session_state.last_refresh).total_seconds()
    st.caption(f"Last refresh: {int(refresh_ago // 60)}m {int(refresh_ago % 60)}s ago")

    if st.button("🔄 Refresh All Data", key="btn_refresh"):
        st.session_state.weather_data = None
        st.session_state.terrain_grid = None
        st.session_state.road_graph = None
        st.session_state.data_loaded = False
        st.session_state.last_refresh = datetime.utcnow()
        st.rerun()

    st.markdown("<hr style='border-color:#0f3460; margin:16px 0;'>", unsafe_allow_html=True)

    # ── Quick Settings ───────────────────────────────────────────────────────
    st.markdown("### ⚙️ Quick Settings")
    st.session_state.lambda_weight = st.slider(
        "λ Hazard Weight", min_value=0.5, max_value=5.0,
        value=st.session_state.lambda_weight, step=0.1, key="sb_lambda",
    )
    st.session_state.risk_threshold_low = st.slider(
        "Low→Moderate threshold", 0.1, 0.5,
        value=st.session_state.risk_threshold_low, step=0.05, key="sb_thr_low",
    )
    st.session_state.risk_threshold_high = st.slider(
        "Moderate→High threshold", 0.4, 0.8,
        value=st.session_state.risk_threshold_high, step=0.05, key="sb_thr_high",
    )

    st.markdown("<hr style='border-color:#0f3460; margin:16px 0;'>", unsafe_allow_html=True)

    # ── Disclaimer ──────────────────────────────────────────────────────────
    with st.expander("⚠️ Disclaimer", expanded=False):
        st.markdown("""
        **PROTOTYPE — NOT FOR OPERATIONAL USE**

        This system is a research and demonstration prototype.
        Predictions are based on simplified hydrological models
        and publicly available APIs.

        - Routes may be inaccurate or unsafe.
        - Risk scores are indicative, not authoritative.
        - Always follow official government advisories.
        - Verify all data with qualified emergency professionals.

        *Data: Open-Meteo · OpenStreetMap · Open-Topo-Data (SRTM)*
        """)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800)
def load_weather_data(lat: float, lon: float) -> dict:
    """Fetch weather data; falls back to synthetic demo data on any failure."""
    try:
        from src.data.weather import get_weather

        record = get_weather(lat, lon)
        if record.is_stale:
            raise RuntimeError("Open-Meteo returned a stale weather record")
        data = {
            "rainfall_mm_hr": record.rainfall_mm_hr,
            "rainfall_24h_mm": record.rainfall_24h_mm,
            "rainfall_72h_mm": record.rainfall_72h_mm,
            "forecast_1h_mm": record.forecast_1h_mm,
            "forecast_3h_mm": record.forecast_3h_mm,
            "forecast_6h_mm": record.forecast_6h_mm,
            "temperature_c": record.temperature_c,
            "humidity_pct": record.humidity_pct,
            "wind_speed_ms": record.wind_speed_ms,
            "imd_alert": _imd_alert_level(record.rainfall_mm_hr),
            "source": record.source,
            "fetched_at": record.fetched_at.isoformat(),
            "is_stale": False,
        }
        return data
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        st.warning(f"Live weather unavailable; using demo data ({exc}).")

    # Demo / synthetic fallback
    rng = np.random.default_rng(int(lat * 100 + lon * 100) % (2**31))
    base_rain = float(rng.uniform(8, 55))
    return {
        "rainfall_mm_hr": round(base_rain, 1),
        "rainfall_24h_mm": round(base_rain * 4.7, 1),
        "rainfall_72h_mm": round(base_rain * 7.9, 1),
        "forecast_1h_mm": round(base_rain * rng.uniform(0.8, 1.4), 1),
        "forecast_3h_mm": round(base_rain * rng.uniform(2.0, 3.5), 1),
        "forecast_6h_mm": round(base_rain * rng.uniform(3.5, 5.5), 1),
        "temperature_c": round(float(rng.uniform(26, 32)), 1),
        "humidity_pct": round(float(rng.uniform(80, 98)), 1),
        "imd_alert": _imd_alert_level(base_rain),
        "source": "demo",
        "fetched_at": datetime.utcnow().isoformat(),
        "is_stale": False,
    }


def _imd_alert_level(mm_hr: float) -> str:
    if mm_hr < 7.5:
        return "Green"
    elif mm_hr < 35.5:
        return "Yellow"
    elif mm_hr < 64.5:
        return "Orange"
    elif mm_hr < 115.5:
        return "Red"
    return "Red+"


def get_risk_color(score: float) -> str:
    if score < 0.3:
        return "#2ecc71"
    elif score < 0.6:
        return "#f39c12"
    elif score < 0.8:
        return "#e74c3c"
    return "#8e44ad"


def get_risk_label(score: float) -> str:
    if score < 0.3:
        return "Low"
    elif score < 0.6:
        return "Moderate"
    elif score < 0.8:
        return "High"
    return "Critical"


def get_risk_pill_class(score: float) -> str:
    if score < 0.3:
        return "pill-safe"
    elif score < 0.6:
        return "pill-moderate"
    elif score < 0.8:
        return "pill-high"
    return "pill-critical"


def compute_flood_risk(
    lat: float, lon: float,
    weather: dict | None,
    elevation_m: float = 5.0,
) -> dict:
    """
    Composite risk score [0–1] based on rainfall, terrain, historical patterns,
    and drainage capacity (simplified model).
    """
    if weather is None:
        weather = {}

    # Rainfall sub-score (0–1)
    rain_1h = weather.get("rainfall_mm_hr", 0.0)
    rain_24 = weather.get("rainfall_24h_mm", 0.0)
    rainfall_risk = min(1.0, (rain_1h / 64.5) * 0.5 + (rain_24 / 200.0) * 0.5)

    # Terrain sub-score (lower elevation → higher risk)
    terrain_risk = max(0.0, min(1.0, 1.0 - elevation_m / 20.0))

    # Historical: Mumbai-specific coastal proximity proxy
    dist_coast_deg = abs(lon - 72.82)
    historical_risk = max(0.0, min(1.0, 0.6 - dist_coast_deg * 3.0))

    # Drainage (simplified — urban areas drain poorly during heavy rain)
    drainage_risk = min(1.0, rain_1h / 50.0 * 0.7 + 0.1)

    lam = st.session_state.get("lambda_weight", 2.0)
    weights = {"rainfall": 0.45, "terrain": 0.25, "historical": 0.15, "drainage": 0.15}
    raw = (
        weights["rainfall"] * rainfall_risk
        + weights["terrain"] * terrain_risk
        + weights["historical"] * historical_risk
        + weights["drainage"] * drainage_risk
    )
    score = float(np.clip(raw * (lam / 2.0), 0.0, 1.0))

    return {
        "score": round(score, 3),
        "label": get_risk_label(score),
        "color": get_risk_color(score),
        "rainfall_risk": round(rainfall_risk, 3),
        "terrain_risk": round(terrain_risk, 3),
        "historical_risk": round(historical_risk, 3),
        "drainage_risk": round(drainage_risk, 3),
        "breakdown": {
            "Rainfall": rainfall_risk,
            "Terrain": terrain_risk,
            "Historical": historical_risk,
            "Drainage": drainage_risk,
        },
    }


def create_main_map(
    study_area: dict,
    weather_data: dict | None,
    incidents: list,
    shelters: list,
    route=None,
    show_risk: bool = True,
    show_shelters_flag: bool = True,
    show_incidents_flag: bool = True,
    flood_risk_score: float = 0.0,
):
    """Build and return a Folium map with all requested layers."""
    import folium
    from folium import plugins as fp

    clat = study_area["lat"]
    clon = study_area["lon"]
    radius_km = study_area["radius_km"]

    fmap = folium.Map(
        location=[clat, clon],
        zoom_start=12,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # ── Risk overlay circle ──────────────────────────────────────────────────
    if show_risk:
        risk_color = get_risk_color(flood_risk_score)
        # Main semi-transparent risk zone
        folium.Circle(
            location=[clat, clon],
            radius=radius_km * 1000,
            color=risk_color,
            fill=True,
            fill_color=risk_color,
            fill_opacity=0.12,
            weight=2,
            opacity=0.5,
            tooltip=f"Risk Zone — {get_risk_label(flood_risk_score)} ({flood_risk_score:.2f})",
        ).add_to(fmap)

        # Gradient sample points for visual variety
        rng = np.random.default_rng(42)
        n_points = 25
        angles = rng.uniform(0, 360, n_points)
        distances = rng.uniform(0.1, 0.85, n_points)
        for angle, dist in zip(angles, distances):
            rad = np.radians(angle)
            dlat = dist * radius_km / 111.0 * np.cos(rad)
            dlon = dist * radius_km / (111.0 * np.cos(np.radians(clat))) * np.sin(rad)
            local_score = np.clip(flood_risk_score + rng.uniform(-0.18, 0.18), 0.0, 1.0)
            folium.CircleMarker(
                location=[clat + dlat, clon + dlon],
                radius=8,
                color=get_risk_color(float(local_score)),
                fill=True,
                fill_color=get_risk_color(float(local_score)),
                fill_opacity=0.35,
                weight=0,
                tooltip=f"Local risk: {float(local_score):.2f}",
            ).add_to(fmap)

    # ── Shelter markers ──────────────────────────────────────────────────────
    if show_shelters_flag:
        for s in shelters:
            occ_pct = int(s["current_occupancy"] / s["capacity"] * 100)
            status_color = (
                "#2ecc71" if s["status"] == "Open"
                else "#f39c12" if s["status"] == "Near Capacity"
                else "#e74c3c"
            )
            popup_html = f"""
            <div style='min-width:180px;font-family:sans-serif;font-size:0.85rem'>
                <b>{s['name']}</b><br>
                <span style='color:{status_color};font-weight:600'>{s['status']}</span><br>
                Type: {s['type']}<br>
                Capacity: {s['current_occupancy']}/{s['capacity']} ({occ_pct}%)<br>
                Flood risk: {get_risk_label(s['flood_risk_score'])}<br>
                Facilities: {', '.join(s['facilities'])}<br>
                📞 {s['contact']}
            </div>
            """
            folium.Marker(
                location=[s["lat"], s["lon"]],
                popup=folium.Popup(popup_html, max_width=240),
                tooltip=f"🏠 {s['name']}",
                icon=folium.Icon(color="green", icon="home", prefix="fa"),
            ).add_to(fmap)

    # ── Incident markers ─────────────────────────────────────────────────────
    if show_incidents_flag:
        sev_colors = {1: "lightblue", 2: "blue", 3: "orange", 4: "red", 5: "darkred"}
        for inc in incidents:
            if inc["status"] == "Resolved":
                continue
            reported_dt = datetime.fromisoformat(inc["reported_at"])
            age_min = int((datetime.utcnow() - reported_dt).total_seconds() / 60)
            popup_html = f"""
            <div style='min-width:200px;font-family:sans-serif;font-size:0.85rem'>
                <b>{inc['id']}</b> — Severity {inc['severity']}/5<br>
                People: {inc['people_count']}
                {'👴' if inc['elderly'] else ''}
                {'👶' if inc['children'] else ''}
                {'🏥' if inc['medical'] else ''}<br>
                Status: <b>{inc['status']}</b><br>
                Priority: {inc['priority_score']:.2f}<br>
                Reported: {age_min}m ago<br>
                <i>{inc['description']}</i>
            </div>
            """
            folium.Marker(
                location=[inc["lat"], inc["lon"]],
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"🚨 {inc['id']} (Sev {inc['severity']})",
                icon=folium.Icon(
                    color=sev_colors.get(inc["severity"], "red"),
                    icon="exclamation-triangle",
                    prefix="fa",
                ),
            ).add_to(fmap)

    # ── Route polyline ───────────────────────────────────────────────────────
    if route is not None:
        segments = route.get("segments", [])
        if segments:
            for seg in segments:
                color = "#e74c3c" if seg.get("flooded") else "#3498db"
                folium.PolyLine(
                    locations=seg["coords"],
                    color=color,
                    weight=5,
                    opacity=0.85,
                    tooltip=seg.get("label", "Route segment"),
                ).add_to(fmap)
        elif route.get("coords"):
            folium.PolyLine(
                locations=route["coords"],
                color="#3498db",
                weight=5,
                opacity=0.85,
                tooltip="Recommended route",
            ).add_to(fmap)
        # Start / end markers
        if route.get("start"):
            folium.Marker(
                location=route["start"],
                tooltip="Start",
                icon=folium.Icon(color="green", icon="play", prefix="fa"),
            ).add_to(fmap)
        if route.get("end"):
            folium.Marker(
                location=route["end"],
                tooltip="Destination",
                icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
            ).add_to(fmap)

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_html = """
    <div style="
        position:fixed; bottom:30px; left:30px; z-index:9999;
        background:rgba(26,26,46,0.93); color:#fff;
        padding:12px 16px; border-radius:8px; font-size:0.78rem;
        border:1px solid #0f3460; min-width:150px;">
        <b style='font-size:0.85rem;'>🗺️ Legend</b><br>
        <span style='color:#2ecc71;'>●</span> Low Risk<br>
        <span style='color:#f39c12;'>●</span> Moderate Risk<br>
        <span style='color:#e74c3c;'>●</span> High Risk<br>
        <span style='color:#8e44ad;'>●</span> Critical Risk<br>
        <hr style='border-color:#0f3460;margin:5px 0'>
        <span style='color:#2ecc71;'>🏠</span> Shelter<br>
        <span style='color:#e74c3c;'>⚠</span> Incident<br>
        <span style='color:#3498db;'>━</span> Route (Safe)<br>
        <span style='color:#e74c3c;'>━</span> Route (Flooded)
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(fmap)

    return fmap


def run_demo_simulation(
    scenario_rainfall: float,
    duration_hours: float,
    land_cover: str,
    center_lat: float,
    center_lon: float,
    radius_km: float,
) -> dict:
    """Synthetic 20×20 grid flood simulation using numpy."""
    GRID = 20
    land_cn = {"Urban": 88, "Suburban": 79, "Rural": 65, "Forest": 55}
    cn = land_cn.get(land_cover, 79)
    S = 25400.0 / cn - 254.0  # SCS retention parameter (mm)

    # Elevation grid (m above datum) — concave bowl shape
    x = np.linspace(-1, 1, GRID)
    y = np.linspace(-1, 1, GRID)
    xx, yy = np.meshgrid(x, y)
    elev = 5.0 + 8.0 * (xx**2 + yy**2) - 3.0 * np.random.default_rng(7).random((GRID, GRID))

    lat_arr = np.linspace(center_lat - radius_km / 111.0, center_lat + radius_km / 111.0, GRID)
    lon_arr = np.linspace(center_lon - radius_km / 111.0, center_lon + radius_km / 111.0, GRID)

    dt_min = 15  # timestep in minutes
    n_steps = int(duration_hours * 60 / dt_min)
    water_depth = np.zeros((GRID, GRID))
    time_steps = []
    first_flood_min = None

    for step in range(1, n_steps + 1):
        elapsed_min = step * dt_min
        # Rainfall in this timestep (mm)
        rain_mm = scenario_rainfall * (dt_min / 60.0)
        # SCS runoff (mm)
        Ia = 0.2 * S
        P = rain_mm
        if P > Ia:
            runoff = ((P - Ia) ** 2) / (P - Ia + S)
        else:
            runoff = 0.0
        # Drain fraction (higher elev drains faster)
        drain = np.clip(0.05 + 0.03 * (elev - elev.min()) / (elev.ptp() + 1e-6), 0, 0.12)
        water_depth = np.maximum(0.0, water_depth + runoff - water_depth * drain)
        # Accumulate in low spots
        water_depth = np.maximum(0.0, water_depth + runoff * 0.1 * (1.0 - np.clip(elev / 15.0, 0, 1)))
        flooded_cells = np.sum(water_depth > 0.1)
        flooded_area = flooded_cells * (radius_km * 2 / GRID) ** 2
        if first_flood_min is None and flooded_cells > 0:
            first_flood_min = elapsed_min
        if step % 2 == 0 or step == n_steps:
            time_steps.append({
                "elapsed_min": elapsed_min,
                "flooded_area_km2": round(float(flooded_area), 3),
                "max_depth_m": round(float(water_depth.max()), 3),
                "avg_depth_m": round(float(water_depth[water_depth > 0.1].mean()) if flooded_cells > 0 else 0.0, 3),
            })

    final_grid = water_depth.copy()
    total_flooded = np.sum(final_grid > 0.1) * (radius_km * 2 / GRID) ** 2

    return {
        "time_steps": time_steps,
        "final_grid": final_grid.tolist(),
        "lats": lat_arr.tolist(),
        "lons": lon_arr.tolist(),
        "summary": {
            "total_flooded_km2": round(float(total_flooded), 2),
            "max_depth_m": round(float(final_grid.max()), 2),
            "time_to_first_flood_min": first_flood_min,
            "cn": cn,
            "S_mm": round(S, 1),
            "rainfall_mm_hr": scenario_rainfall,
            "duration_hr": duration_hours,
            "land_cover": land_cover,
        },
    }


def compute_demo_routes(
    start_lat: float, start_lon: float,
    end_lat: float, end_lon: float,
    flood_risk: float,
) -> dict:
    """Generate three synthetic route alternatives."""
    dist_deg = np.hypot(end_lat - start_lat, end_lon - start_lon)
    dist_km = dist_deg * 111.0

    # Interpolate route coordinates
    def _interp_coords(slat, slon, elat, elon, n=8, jitter=0.002):
        rng = np.random.default_rng(13)
        lats = np.linspace(slat, elat, n) + rng.uniform(-jitter, jitter, n)
        lons = np.linspace(slon, elon, n) + rng.uniform(-jitter, jitter, n)
        lats[[0, -1]] = [slat, elat]
        lons[[0, -1]] = [slon, elon]
        return list(zip(lats.tolist(), lons.tolist()))

    baseline_coords = _interp_coords(start_lat, start_lon, end_lat, end_lon, jitter=0.001)
    flood_aware_coords = _interp_coords(start_lat, start_lon, end_lat, end_lon, n=10, jitter=0.006)

    baseline_time = dist_km / 30.0 * 60  # 30 km/h
    fa_time = dist_km / 22.0 * 60 * (1 + flood_risk * 0.5)  # slower due to detours
    fast_time = dist_km / 40.0 * 60  # fastest but riskier

    # Segment the flood-aware route — mark middle portion as flooded if risk high
    fa_segments = []
    all_coords = flood_aware_coords
    n = len(all_coords)
    if flood_risk > 0.5:
        fa_segments = [
            {"coords": all_coords[:n // 3 + 1], "flooded": False, "label": "Safe segment"},
            {"coords": all_coords[n // 3: 2 * n // 3 + 1], "flooded": True, "label": "⚠️ Flood risk zone"},
            {"coords": all_coords[2 * n // 3:], "flooded": False, "label": "Safe segment"},
        ]
    else:
        fa_segments = [{"coords": all_coords, "flooded": False, "label": "Flood-aware route"}]

    return {
        "baseline": {
            "name": "Baseline (Shortest)",
            "distance_km": round(dist_km, 2),
            "time_min": round(baseline_time, 1),
            "flood_exposure": round(flood_risk * 0.95, 2),
            "safety_score": round(1 - flood_risk * 0.95, 2),
            "coords": baseline_coords,
            "segments": [{"coords": baseline_coords, "flooded": False, "label": "Baseline route"}],
        },
        "flood_aware": {
            "name": "Flood-Aware (Recommended)",
            "distance_km": round(dist_km * 1.18, 2),
            "time_min": round(fa_time, 1),
            "flood_exposure": round(flood_risk * 0.25, 2),
            "safety_score": round(1 - flood_risk * 0.25, 2),
            "coords": flood_aware_coords,
            "segments": fa_segments,
        },
        "fastest": {
            "name": "Fastest Available",
            "distance_km": round(dist_km * 0.96, 2),
            "time_min": round(fast_time, 1),
            "flood_exposure": round(flood_risk * 0.88, 2),
            "safety_score": round(1 - flood_risk * 0.88, 2),
            "coords": _interp_coords(start_lat, start_lon, end_lat, end_lon, n=6, jitter=0.0008),
            "segments": [],
        },
        "start": [start_lat, start_lon],
        "end": [end_lat, end_lon],
    }


def format_time_delta(minutes: float) -> str:
    """Format minutes as 'Xh Ym'."""
    h = int(minutes // 60)
    m = int(minutes % 60)
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m"


def _generate_rainfall_series(weather_data: dict | None) -> pd.DataFrame:
    """Generate a 72h + 6h synthetic rainfall time-series."""
    rng = np.random.default_rng(77)
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    times = [now - timedelta(hours=72 - i) for i in range(79)]
    base = weather_data.get("rainfall_mm_hr", 18.5) if weather_data else 18.5
    # Build plausible monsoon pattern
    vals = []
    cur = base * 0.3
    for i, t in enumerate(times):
        # forecast portion
        is_forecast = i >= 72
        if i % 24 < 6:
            cur = cur * 0.7 + rng.uniform(0, base * 0.5) * 0.3
        elif i % 24 < 12:
            cur = cur * 0.5 + rng.uniform(base * 0.3, base * 1.2) * 0.5
        else:
            cur = cur * 0.6 + rng.uniform(base * 0.1, base * 0.8) * 0.4
        cur = float(np.clip(cur + rng.uniform(-3, 3), 0, 130))
        vals.append(cur)
    df = pd.DataFrame({"datetime": times, "rainfall_mm_hr": vals})
    df["is_forecast"] = df["datetime"] > now - timedelta(hours=1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Command Center",
    "📊 Analytics & Forecast",
    "🌊 Scenario Simulator",
    "🚑 Rescue Operations",
])

# Pre-load weather data if not already loaded
if st.session_state.weather_data is None:
    with st.spinner("Fetching weather data …"):
        st.session_state.weather_data = load_weather_data(
            st.session_state.study_area["lat"],
            st.session_state.study_area["lon"],
        )
        st.session_state.data_loaded = True

weather = st.session_state.weather_data
risk = compute_flood_risk(
    st.session_state.study_area["lat"],
    st.session_state.study_area["lon"],
    weather,
    elevation_m=4.5,
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — COMMAND CENTER
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    map_col, ctrl_col = st.columns([3, 1])

    with map_col:
        # Data freshness banner
        data_source = weather.get("source", "demo")
        if data_source == "demo":
            st.markdown(
                '<div class="warning-banner">⚠️ <b>Demo Mode:</b> Displaying synthetic data — live API unavailable. '
                'All values are illustrative only.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="live-indicator"><span class="live-dot"></span> LIVE DATA</div>',
                unsafe_allow_html=True,
            )

        # Build route for map
        active_route = None
        if (
            st.session_state.route_comparison is not None
            and st.session_state.show_route
        ):
            active_route = st.session_state.route_comparison["flood_aware"]

        # Draw map
        fmap = create_main_map(
            study_area=st.session_state.study_area,
            weather_data=weather,
            incidents=st.session_state.incidents,
            shelters=st.session_state.shelters,
            route=active_route,
            show_risk=st.session_state.show_flood_risk_layer,
            show_shelters_flag=st.session_state.show_shelters,
            show_incidents_flag=st.session_state.show_incidents,
            flood_risk_score=risk["score"],
        )

        try:
            from streamlit_folium import st_folium
            st_folium(fmap, height=600, use_container_width=True)
        except ImportError:
            st.warning("streamlit-folium not installed. Install with: pip install streamlit-folium")
            st.markdown("*(Map unavailable — install streamlit-folium to view)*")

        # Metrics row
        mc1, mc2, mc3, mc4 = st.columns(4)
        imd = weather.get("imd_alert", "Green")
        imd_colors = {
            "Green": "#2ecc71", "Yellow": "#f1c40f",
            "Orange": "#f39c12", "Red": "#e74c3c", "Red+": "#8e44ad",
        }
        imd_color = imd_colors.get(imd, "#2ecc71")
        active_inc = sum(1 for i in st.session_state.incidents if i["status"] == "Active")
        open_shelters = sum(1 for s in st.session_state.shelters if s["status"] == "Open")

        with mc1:
            st.markdown(f"""
            <div class="metric-card metric-yellow">
                <div class="metric-label">🌧️ Rainfall</div>
                <div class="metric-value">{weather.get('rainfall_mm_hr', 0):.1f}</div>
                <div class="metric-delta">mm/hr  |  24h: {weather.get('rainfall_24h_mm', 0):.0f} mm</div>
            </div>""", unsafe_allow_html=True)
        with mc2:
            risk_css = ["metric-green", "metric-yellow", "metric-red", "metric-purple"]
            risk_idx = min(int(risk["score"] / 0.25), 3)
            st.markdown(f"""
            <div class="metric-card {risk_css[risk_idx]}">
                <div class="metric-label">⚠️ Flood Risk</div>
                <div class="metric-value">{risk['score']:.2f}</div>
                <div class="metric-delta">{risk['label']} — IMD: {imd}</div>
            </div>""", unsafe_allow_html=True)
        with mc3:
            st.markdown(f"""
            <div class="metric-card metric-red">
                <div class="metric-label">🚨 Active Incidents</div>
                <div class="metric-value">{active_inc}</div>
                <div class="metric-delta">of {len(st.session_state.incidents)} total reported</div>
            </div>""", unsafe_allow_html=True)
        with mc4:
            st.markdown(f"""
            <div class="metric-card metric-green">
                <div class="metric-label">🏠 Open Shelters</div>
                <div class="metric-value">{open_shelters}</div>
                <div class="metric-delta">of {len(st.session_state.shelters)} total facilities</div>
            </div>""", unsafe_allow_html=True)

    with ctrl_col:
        st.markdown("### 🎛️ Controls")

        # ── Risk Assessment ──
        with st.expander("⚠️ Risk Assessment", expanded=True):
            import plotly.graph_objects as go
            breakdown = risk["breakdown"]
            fig_risk = go.Figure(go.Bar(
                x=list(breakdown.values()),
                y=list(breakdown.keys()),
                orientation="h",
                marker_color=[
                    get_risk_color(v) for v in breakdown.values()
                ],
                text=[f"{v:.2f}" for v in breakdown.values()],
                textposition="outside",
            ))
            fig_risk.update_layout(
                xaxis=dict(range=[0, 1], title="Score"),
                height=180,
                margin=dict(l=0, r=30, t=10, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(size=11),
            )
            st.plotly_chart(fig_risk, use_container_width=True)

            pill_cls = get_risk_pill_class(risk["score"])
            st.markdown(
                f'Overall risk: <span class="{pill_cls}">{risk["label"]}</span> ({risk["score"]:.3f})',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'IMD Alert: <span style="background:{imd_color};color:#fff;'
                f'padding:1px 8px;border-radius:10px;font-size:0.78rem;font-weight:600;">'
                f'{imd}</span>',
                unsafe_allow_html=True,
            )
            if weather.get("source") == "demo":
                st.markdown(
                    '<span style="color:#f39c12;font-size:0.78rem;">⚠️ Demo data — live API not reached</span>',
                    unsafe_allow_html=True,
                )

        # ── Route Planning ──
        with st.expander("🗺️ Route Planning", expanded=False):
            clat = st.session_state.study_area["lat"]
            clon = st.session_state.study_area["lon"]
            st.markdown("**Start Point**")
            slat = st.number_input("Start Lat", value=clat - 0.04, format="%.4f", key="rp_slat")
            slon = st.number_input("Start Lon", value=clon - 0.03, format="%.4f", key="rp_slon")
            st.markdown("**End Point**")
            elat = st.number_input("End Lat", value=clat + 0.04, format="%.4f", key="rp_elat")
            elon = st.number_input("End Lon", value=clon + 0.03, format="%.4f", key="rp_elon")

            if st.button("🗺️ Compute Routes", key="btn_compute_routes"):
                with st.spinner("Computing routes …"):
                    rc = compute_demo_routes(slat, slon, elat, elon, risk["score"])
                    st.session_state.route_comparison = rc
                st.success("Routes computed!")

            if st.session_state.route_comparison:
                rc = st.session_state.route_comparison
                route_names = ["baseline", "flood_aware", "fastest"]
                rows = []
                for rn in route_names:
                    r = rc[rn]
                    rows.append({
                        "Route": r["name"],
                        "Dist (km)": r["distance_km"],
                        "Time": format_time_delta(r["time_min"]),
                        "Flood Exp.": f"{r['flood_exposure']:.2f}",
                        "Safety": f"{r['safety_score']:.2f}",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.caption("✅ Flood-aware route selected on map")

        # ── Map Layers ──
        with st.expander("🗂️ Map Layers", expanded=True):
            st.session_state.show_flood_risk_layer = st.checkbox(
                "Show Flood Risk Layer", value=st.session_state.show_flood_risk_layer, key="chk_risk"
            )
            st.session_state.show_shelters = st.checkbox(
                "Show Shelters", value=st.session_state.show_shelters, key="chk_shelters"
            )
            st.session_state.show_incidents = st.checkbox(
                "Show Incidents", value=st.session_state.show_incidents, key="chk_incidents"
            )
            st.session_state.show_route = st.checkbox(
                "Show Route", value=st.session_state.show_route, key="chk_route"
            )
            st.caption("Changes apply on next rerun. Press R to reload.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — ANALYTICS & FORECAST
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    import plotly.graph_objects as go
    import plotly.express as px

    st.markdown("## 📊 Analytics & Forecast")

    # ── Rainfall Timeline ────────────────────────────────────────────────────
    rain_df = _generate_rainfall_series(weather)
    historical = rain_df[~rain_df["is_forecast"]]
    forecast = rain_df[rain_df["is_forecast"]]

    fig_rain = go.Figure()
    fig_rain.add_trace(go.Scatter(
        x=historical["datetime"], y=historical["rainfall_mm_hr"],
        mode="lines", name="Observed",
        line=dict(color="#3498db", width=2),
        fill="tozeroy", fillcolor="rgba(52,152,219,0.15)",
    ))
    fig_rain.add_trace(go.Scatter(
        x=forecast["datetime"], y=forecast["rainfall_mm_hr"],
        mode="lines", name="Forecast",
        line=dict(color="#e67e22", width=2, dash="dash"),
        fill="tozeroy", fillcolor="rgba(230,126,34,0.10)",
    ))
    # IMD threshold lines
    for level, val, color in [
        ("Light (7.5)", 7.5, "#2ecc71"),
        ("Moderate (35.5)", 35.5, "#f39c12"),
        ("Heavy (64.5)", 64.5, "#e74c3c"),
        ("Extreme (115.5)", 115.5, "#8e44ad"),
    ]:
        fig_rain.add_hline(
            y=val, line_dash="dot", line_color=color, line_width=1.5,
            annotation_text=level, annotation_position="right",
            annotation_font_size=10, annotation_font_color=color,
        )
    fig_rain.update_layout(
        title="Rainfall — Last 72 Hours + 6-Hour Forecast",
        xaxis_title="Date / Time (UTC)",
        yaxis_title="Rainfall (mm/hr)",
        height=350,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
        hovermode="x unified",
        margin=dict(l=10, r=80, t=50, b=30),
    )
    st.plotly_chart(fig_rain, use_container_width=True)

    # ── Forecast Panel ───────────────────────────────────────────────────────
    fc1, fc2 = st.columns(2)

    with fc1:
        st.markdown("#### 🔭 Short-Range Forecast")
        fc_labels = ["1-Hour", "3-Hour", "6-Hour"]
        fc_vals = [
            weather.get("forecast_1h_mm", 22),
            weather.get("forecast_3h_mm", 58),
            weather.get("forecast_6h_mm", 95),
        ]
        fc_colors = [get_risk_color(v / 115.0) for v in fc_vals]
        fig_fc = go.Figure(go.Bar(
            x=fc_labels, y=fc_vals,
            marker_color=fc_colors,
            text=[f"{v:.1f} mm" for v in fc_vals],
            textposition="outside",
        ))
        fig_fc.add_hline(y=35.5, line_dash="dot", line_color="#f39c12",
                         annotation_text="Moderate threshold")
        fig_fc.add_hline(y=64.5, line_dash="dot", line_color="#e74c3c",
                         annotation_text="Heavy threshold")
        fig_fc.update_layout(
            yaxis_title="Cumulative Rainfall (mm)",
            height=280,
            margin=dict(l=10, r=10, t=20, b=20),
        )
        st.plotly_chart(fig_fc, use_container_width=True)

    with fc2:
        st.markdown("#### 🧩 Risk Component Breakdown")
        cats = list(risk["breakdown"].keys())
        vals_r = [risk["breakdown"][c] for c in cats]
        fig_br = go.Figure()
        fig_br.add_trace(go.Bar(
            x=cats, y=vals_r,
            marker_color=[get_risk_color(v) for v in vals_r],
            text=[f"{v:.2f}" for v in vals_r],
            textposition="outside",
        ))
        fig_br.update_layout(
            yaxis=dict(range=[0, 1.1], title="Score"),
            height=280,
            margin=dict(l=10, r=10, t=20, b=20),
        )
        st.plotly_chart(fig_br, use_container_width=True)

    # ── Data Quality Panel ───────────────────────────────────────────────────
    st.markdown("#### 📋 Data Source Status")
    fetched_str = weather.get("fetched_at", datetime.utcnow().isoformat())
    try:
        fetched_dt = datetime.fromisoformat(fetched_str)
        age_min = int((datetime.utcnow() - fetched_dt).total_seconds() / 60)
        freshness_str = f"{age_min}m ago"
    except Exception:
        freshness_str = "Unknown"

    dq_df = pd.DataFrame([
        {
            "Source": "Open-Meteo API",
            "Data": "Rainfall, Temperature, Humidity",
            "Last Updated": fetched_str[:16],
            "Freshness": freshness_str,
            "Status": "✅ Live" if weather.get("source") == "api" else "⚠️ Demo",
            "Coverage": "Global (hourly)",
        },
        {
            "Source": "Open-Topo-Data (SRTM)",
            "Data": "DEM / Elevation",
            "Last Updated": "2000-02-11",
            "Freshness": "Archive",
            "Status": "✅ Cached" if st.session_state.terrain_grid else "⬜ Not loaded",
            "Coverage": "±60° latitude",
        },
        {
            "Source": "OpenStreetMap (Overpass)",
            "Data": "Road Network",
            "Last Updated": "Rolling",
            "Freshness": "Real-time",
            "Status": "✅ Cached" if st.session_state.road_graph else "⬜ Not loaded",
            "Coverage": "Global",
        },
        {
            "Source": "IMD (Manual)",
            "Data": "Alert Level",
            "Last Updated": fetched_str[:16],
            "Freshness": freshness_str,
            "Status": "⚠️ Estimated",
            "Coverage": "India",
        },
    ])
    st.dataframe(dq_df, use_container_width=True, hide_index=True)

    # ── IMD Classification Table ─────────────────────────────────────────────
    st.markdown("#### 🌈 IMD Rainfall Classification")
    imd_df = pd.DataFrame([
        {"Level": "🟢 Green",  "Threshold": "< 7.5 mm/hr",   "Category": "Light Rain",    "Action": "Monitor"},
        {"Level": "🟡 Yellow", "Threshold": "7.5–35.5 mm/hr", "Category": "Moderate Rain", "Action": "Advisory issued"},
        {"Level": "🟠 Orange", "Threshold": "35.5–64.5 mm/hr","Category": "Heavy Rain",    "Action": "Alert — avoid low areas"},
        {"Level": "🔴 Red",    "Threshold": "64.5–115.5 mm/hr","Category": "Very Heavy",   "Action": "Warning — evacuate risk zones"},
        {"Level": "🟣 Red+",   "Threshold": "> 115.5 mm/hr",  "Category": "Extremely Heavy","Action": "Emergency — immediate evacuation"},
    ])
    st.dataframe(imd_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — SCENARIO SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    import plotly.graph_objects as go
    import plotly.express as px

    st.markdown("## 🌊 Scenario Simulator")
    st.markdown(
        '<div class="info-banner">ℹ️ Run synthetic flood simulations using the SCS Curve Number method. '
        'Results are illustrative — not a certified hydraulic model.</div>',
        unsafe_allow_html=True,
    )

    sim_ctrl, sim_viz = st.columns([1, 2])

    land_covers = ["Urban", "Suburban", "Rural", "Forest"]
    cn_map = {"Urban": 88, "Suburban": 79, "Rural": 65, "Forest": 55}

    with sim_ctrl:
        st.markdown("### ⚙️ Scenario Parameters")
        scenario_name = st.text_input("Scenario Name", value="My Scenario", key="sim_name")
        sim_rain = st.slider(
            "Rainfall Intensity (mm/hr)", 0.0, 200.0,
            value=st.session_state.scenario_rainfall, step=1.0, key="sim_rain"
        )
        sim_duration = st.slider(
            "Storm Duration (hours)", 1.0, 24.0,
            value=st.session_state.scenario_duration, step=0.5, key="sim_dur"
        )
        sim_lc = st.selectbox(
            "Land Cover Type",
            options=land_covers,
            index=land_covers.index(st.session_state.scenario_land_cover)
                if st.session_state.scenario_land_cover in land_covers else 1,
            key="sim_lc",
        )
        cn_val = cn_map.get(sim_lc, 79)
        st.metric("SCS Curve Number (CN)", cn_val, help="Higher CN = more runoff")

        run_sim = st.button("▶️ Run Simulation", key="btn_run_sim", type="primary")

        st.markdown("---")
        st.markdown("**📦 Scenario Presets**")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            if st.button("🌤️ Light Rain", key="pre_light"):
                st.session_state.scenario_rainfall = 5.0
                st.session_state.scenario_duration = 3.0
                st.session_state.scenario_land_cover = "Rural"
                st.rerun()
            if st.button("🌧️ Heavy Monsoon", key="pre_heavy"):
                st.session_state.scenario_rainfall = 75.0
                st.session_state.scenario_duration = 12.0
                st.session_state.scenario_land_cover = "Urban"
                st.rerun()
        with col_p2:
            if st.button("🌦️ Moderate Storm", key="pre_mod"):
                st.session_state.scenario_rainfall = 35.0
                st.session_state.scenario_duration = 6.0
                st.session_state.scenario_land_cover = "Suburban"
                st.rerun()
            if st.button("⛈️ Extreme Event", key="pre_extreme"):
                st.session_state.scenario_rainfall = 130.0
                st.session_state.scenario_duration = 18.0
                st.session_state.scenario_land_cover = "Urban"
                st.rerun()

    with sim_viz:
        if run_sim:
            st.session_state.scenario_rainfall = sim_rain
            st.session_state.scenario_duration = sim_duration
            st.session_state.scenario_land_cover = sim_lc
            with st.spinner(f"Running '{scenario_name}' …"):
                result = run_demo_simulation(
                    sim_rain, sim_duration, sim_lc,
                    st.session_state.study_area["lat"],
                    st.session_state.study_area["lon"],
                    st.session_state.study_area["radius_km"],
                )
                st.session_state.simulation_result = result
            st.success("Simulation complete!")

        if st.session_state.simulation_result:
            res = st.session_state.simulation_result
            summ = res["summary"]
            ts = pd.DataFrame(res["time_steps"])

            # Summary metrics
            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Flooded Area", f"{summ['total_flooded_km2']} km²")
            sm2.metric("Max Water Depth", f"{summ['max_depth_m']} m")
            tf = summ["time_to_first_flood_min"]
            sm3.metric("First Flooding", f"{tf} min" if tf else "None")
            sm4.metric("SCS CN", summ["cn"])

            # Time-series chart
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=ts["elapsed_min"], y=ts["flooded_area_km2"],
                mode="lines+markers", name="Flooded Area (km²)",
                line=dict(color="#3498db", width=2),
                fill="tozeroy", fillcolor="rgba(52,152,219,0.15)",
            ))
            fig_ts.add_trace(go.Scatter(
                x=ts["elapsed_min"], y=ts["max_depth_m"],
                mode="lines", name="Max Depth (m)",
                line=dict(color="#e74c3c", width=2, dash="dash"),
                yaxis="y2",
            ))
            fig_ts.update_layout(
                title=f"Flood Progression — {scenario_name}",
                xaxis_title="Elapsed Time (minutes)",
                yaxis_title="Flooded Area (km²)",
                yaxis2=dict(title="Max Depth (m)", overlaying="y", side="right"),
                height=280,
                margin=dict(l=10, r=60, t=50, b=30),
                hovermode="x unified",
                legend=dict(x=0.01, y=0.99),
            )
            st.plotly_chart(fig_ts, use_container_width=True)

            # Final depth heatmap
            grid = np.array(res["final_grid"])
            fig_hm = px.imshow(
                grid,
                color_continuous_scale=[
                    [0.0, "#e8f4f8"],
                    [0.1, "#3498db"],
                    [0.5, "#e67e22"],
                    [1.0, "#8e44ad"],
                ],
                labels=dict(color="Depth (m)"),
                title="Final Water Depth Grid",
                aspect="equal",
            )
            fig_hm.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_hm, use_container_width=True)

            st.markdown(
                '<div class="warning-banner">⚠️ <b>LIMITATIONS:</b> This uses a simplified SCS Curve Number model '
                'on a 20×20 synthetic grid. It does not account for channel hydraulics, sewer networks, '
                'real DEM data, or spatially-varying land cover. Results are for educational purposes only.'
                '</div>',
                unsafe_allow_html=True,
            )

            # Export
            csv_out = ts.to_csv(index=False)
            st.download_button(
                "⬇️ Download Simulation CSV",
                data=csv_out,
                file_name=f"simulation_{scenario_name.replace(' ', '_')}.csv",
                mime="text/csv",
                key="dl_sim_csv",
            )
        else:
            st.markdown(
                '<div style="text-align:center;padding:60px 0;color:#888;">'
                '<div style="font-size:3rem;">🌊</div>'
                '<div style="font-size:1.1rem;margin-top:10px;">Configure scenario and press <b>▶️ Run Simulation</b></div>'
                '</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — RESCUE OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("## 🚑 Rescue Operations")

    inc_col, she_col, disp_col = st.columns(3)

    # ── Column 1: Incident Management ────────────────────────────────────────
    with inc_col:
        active_count = sum(1 for i in st.session_state.incidents if i["status"] == "Active")
        st.markdown(f"### 🚨 Incidents ({active_count} Active)")

        with st.expander("➕ Report New Incident", expanded=False):
            with st.form("new_incident_form", clear_on_submit=True):
                ni_lat = st.number_input("Latitude", value=st.session_state.study_area["lat"], format="%.4f", key="ni_lat")
                ni_lon = st.number_input("Longitude", value=st.session_state.study_area["lon"], format="%.4f", key="ni_lon")
                ni_sev = st.slider("Severity", 1, 5, 3, key="ni_sev",
                                   help="1=Minor, 2=Low, 3=Moderate, 4=High, 5=Critical")
                sev_labels = {1: "Minor", 2: "Low", 3: "Moderate", 4: "High", 5: "Critical"}
                st.caption(f"Severity: **{sev_labels[ni_sev]}**")
                ni_people = st.number_input("Number of People", 1, 500, 5, key="ni_people")
                ni_elderly = st.checkbox("Elderly present", key="ni_elderly")
                ni_children = st.checkbox("Children present", key="ni_children")
                ni_medical = st.checkbox("Medical needs", key="ni_medical")
                ni_desc = st.text_area("Description", key="ni_desc", max_chars=300)
                submit_inc = st.form_submit_button("📝 Submit Incident")
                if submit_inc:
                    inc_id = f"INC-{100 + len(st.session_state.incidents) + 1}"
                    vuln = (ni_elderly + ni_children + ni_medical) / 3.0
                    p_score = round(min(1.0, (ni_sev / 5.0) * 0.6 + (ni_people / 50.0) * 0.2 + vuln * 0.2), 2)
                    st.session_state.incidents.append({
                        "id": inc_id,
                        "lat": ni_lat, "lon": ni_lon,
                        "severity": ni_sev, "people_count": ni_people,
                        "elderly": ni_elderly, "children": ni_children, "medical": ni_medical,
                        "description": ni_desc,
                        "status": "Active",
                        "reported_at": datetime.utcnow().isoformat(),
                        "priority_score": p_score,
                    })
                    st.success(f"Incident {inc_id} reported (Priority: {p_score:.2f})")

        # Sort incidents by priority
        sorted_incs = sorted(
            st.session_state.incidents,
            key=lambda x: x["priority_score"],
            reverse=True,
        )

        for inc in sorted_incs:
            age_min = int((datetime.utcnow() - datetime.fromisoformat(inc["reported_at"])).total_seconds() / 60)
            p = inc["priority_score"]
            p_color = get_risk_color(p)
            vuln_icons = (
                ("👴" if inc["elderly"] else "")
                + ("👶" if inc["children"] else "")
                + ("🏥" if inc["medical"] else "")
            )
            time_warn = ""
            if inc["status"] == "Active" and age_min > 30:
                time_warn = f'<br><span style="color:#e74c3c;font-size:0.75rem;">⏰ Waiting {age_min}m — needs attention!</span>'

            st.markdown(f"""
            <div class="incident-card" style="border-left-color:{p_color};">
                <b>{inc['id']}</b>
                &nbsp;
                <span style="background:{p_color};color:#fff;padding:1px 7px;border-radius:8px;font-size:0.75rem;">
                    P {p:.2f}
                </span>
                &nbsp; Sev {inc['severity']}/5 &nbsp; {vuln_icons}<br>
                <span style="font-size:0.82rem;color:#555;">{inc['description'][:80]}{'…' if len(inc['description'])>80 else ''}</span>
                <br>
                <span style="font-size:0.78rem;color:#888;">📍 {inc['lat']:.4f}, {inc['lon']:.4f} | {age_min}m ago</span>
                {time_warn}
            </div>
            """, unsafe_allow_html=True)

            new_status = st.selectbox(
                f"Status ({inc['id']})",
                ["Active", "In Progress", "Resolved"],
                index=["Active", "In Progress", "Resolved"].index(inc["status"]),
                key=f"inc_status_{inc['id']}",
            )
            inc["status"] = new_status

    # ── Column 2: Shelter Status ──────────────────────────────────────────────
    with she_col:
        open_count = sum(1 for s in st.session_state.shelters if s["status"] == "Open")
        st.markdown(f"### 🏠 Shelters ({open_count} Open)")

        for shel in st.session_state.shelters:
            occ_pct = shel["current_occupancy"] / shel["capacity"]
            bar_color = "#2ecc71" if occ_pct < 0.7 else "#f39c12" if occ_pct < 0.95 else "#e74c3c"
            status_styles = {
                "Open": "background:#2ecc71;",
                "Near Capacity": "background:#f39c12;",
                "Full": "background:#e74c3c;",
            }
            s_style = status_styles.get(shel["status"], "background:#95a5a6;")
            fac_icons = {
                "food": "🍽️", "water": "💧", "medical": "🏥",
                "power": "⚡", "toilets": "🚻",
            }
            fac_str = " ".join(fac_icons.get(f, f) for f in shel["facilities"])
            risk_lbl = get_risk_label(shel["flood_risk_score"])
            risk_col = get_risk_color(shel["flood_risk_score"])

            st.markdown(f"""
            <div class="shelter-card">
                <b>{shel['name']}</b>
                &nbsp;
                <span style="{s_style}color:#fff;padding:1px 7px;border-radius:8px;font-size:0.75rem;">
                    {shel['status']}
                </span><br>
                <span style="font-size:0.8rem;color:#555;">{shel['type']}</span><br>
                <span style="font-size:0.8rem;">
                    Occupancy: {shel['current_occupancy']}/{shel['capacity']} ({int(occ_pct*100)}%)
                </span>
            </div>
            """, unsafe_allow_html=True)

            st.progress(float(occ_pct), text="")
            st.markdown(
                f'<span style="font-size:0.78rem;color:#555;">Flood risk: '
                f'<span style="color:{risk_col};font-weight:600;">{risk_lbl}</span> '
                f'| Facilities: {fac_str}</span>',
                unsafe_allow_html=True,
            )
            new_occ = st.number_input(
                f"Update occupancy ({shel['id']})",
                min_value=0, max_value=shel["capacity"],
                value=shel["current_occupancy"],
                key=f"she_occ_{shel['id']}",
            )
            shel["current_occupancy"] = new_occ
            if new_occ >= shel["capacity"]:
                shel["status"] = "Full"
            elif new_occ >= int(shel["capacity"] * 0.85):
                shel["status"] = "Near Capacity"
            else:
                shel["status"] = "Open"

    # ── Column 3: Dispatch & Assignment ──────────────────────────────────────
    with disp_col:
        st.markdown("### 📡 Dispatch & Assignment")

        st.markdown("#### 🔍 Route to Nearest Safe Shelter")
        active_incs = [i for i in st.session_state.incidents if i["status"] == "Active"]
        if not active_incs:
            st.info("No active incidents.")
        else:
            inc_options = {f"{i['id']} — Sev {i['severity']} ({i['people_count']} people)": i for i in active_incs}
            sel_inc_key = st.selectbox("Select Incident", list(inc_options.keys()), key="disp_inc_sel")
            sel_inc = inc_options[sel_inc_key]

            if st.button("🔎 Find Nearest Safe Shelter", key="btn_find_shelter"):
                # Find open shelter with lowest flood risk and enough capacity
                available = [
                    s for s in st.session_state.shelters
                    if s["status"] in ("Open", "Near Capacity")
                    and s["current_occupancy"] < s["capacity"]
                ]
                if not available:
                    st.error("No shelters available!")
                else:
                    # Score: distance + flood risk
                    def _shelter_score(s):
                        d = np.hypot(s["lat"] - sel_inc["lat"], s["lon"] - sel_inc["lon"])
                        return d * 0.7 + s["flood_risk_score"] * 0.3

                    best = min(available, key=_shelter_score)
                    dist_km = np.hypot(
                        best["lat"] - sel_inc["lat"],
                        best["lon"] - sel_inc["lon"],
                    ) * 111.0
                    eta_min = dist_km / 20.0 * 60  # 20 km/h emergency speed

                    st.success(f"✅ Nearest safe shelter: **{best['name']}**")
                    st.markdown(f"""
                    | Field | Value |
                    |-------|-------|
                    | Shelter | {best['name']} |
                    | Type | {best['type']} |
                    | Distance | {dist_km:.1f} km |
                    | ETA | {format_time_delta(eta_min)} |
                    | Capacity | {best['current_occupancy']}/{best['capacity']} |
                    | Flood Risk | {get_risk_label(best['flood_risk_score'])} |
                    | Contact | {best['contact']} |
                    """)

                    # Log dispatch
                    log_entry = {
                        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        "incident": sel_inc["id"],
                        "shelter": best["name"],
                        "dist_km": round(dist_km, 2),
                        "eta_min": round(eta_min, 1),
                        "status": "Dispatched",
                    }
                    st.session_state.dispatch_log.append(log_entry)

        # Audit trail
        st.markdown("---")
        st.markdown("#### 📋 Dispatch Audit Log")
        if st.session_state.dispatch_log:
            log_df = pd.DataFrame(st.session_state.dispatch_log)
            st.dataframe(log_df, use_container_width=True, hide_index=True)
            dl_log = log_df.to_csv(index=False)
            st.download_button(
                "⬇️ Export Log CSV",
                data=dl_log,
                file_name="dispatch_log.csv",
                mime="text/csv",
                key="dl_log_csv",
            )
        else:
            st.caption("No dispatches recorded yet.")

        # Quick stats
        st.markdown("---")
        st.markdown("#### 📈 Operations Summary")
        total_cap = sum(s["capacity"] for s in st.session_state.shelters)
        total_occ = sum(s["current_occupancy"] for s in st.session_state.shelters)
        total_people_at_risk = sum(
            i["people_count"] for i in st.session_state.incidents if i["status"] == "Active"
        )
        st.metric("Total Shelter Capacity", f"{total_occ}/{total_cap}")
        st.metric("People in Active Incidents", total_people_at_risk)
        st.metric("Dispatches Made", len(st.session_state.dispatch_log))
        st.metric(
            "Shelter Utilisation",
            f"{int(total_occ / total_cap * 100) if total_cap > 0 else 0}%",
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
model_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
st.markdown(f"""
<div class="footer">
    <b>⚠️ PROTOTYPE</b> — For demonstration and research purposes only.
    Not for use in actual emergency operations. All risk assessments and routes
    must be validated by qualified emergency management professionals.<br><br>
    Data sources:
    <a href="https://open-meteo.com/" target="_blank">Open-Meteo</a> ·
    <a href="https://www.openstreetmap.org/" target="_blank">OpenStreetMap</a> ·
    <a href="https://www.opentopodata.org/" target="_blank">Open-Topo-Data (SRTM)</a><br>
    Model: FloodGuard AI v0.1-proto &nbsp;|&nbsp; Generated: {model_ts}
</div>
""", unsafe_allow_html=True)
