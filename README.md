# Flood Prediction & Rescue Routing

This project is a Streamlit prototype for flood-risk screening, simplified
flood-spread simulation, and hazard-aware rescue route comparison.

## Run locally

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

The application can run in demo mode without API keys. Live weather is
retrieved from Open-Meteo, terrain from Open-Topo-Data, and road data from
OpenStreetMap through the adapters under `src/data`.

## Python API

The main building blocks are:

- `src.flood.risk_model.FloodRiskModel` for the weighted, non-calibrated risk
  index.
- `src.flood.runoff.SCSRunoffModel` for SCS Curve Number runoff estimates.
- `src.flood.simulation.FloodSimulator` for Level 1 and simplified Level 2
  screening simulations.
- `src.routing.graph_builder.FloodAwareGraphBuilder` and
  `src.routing.route_planner.RoutePlanner` for hazard-aware routing.
- `src.evaluation.flood_metrics.FloodPredictionEvaluator` and
  `src.evaluation.route_metrics.compare_routes` for offline evaluation.

## Important limitations

This is a decision-support prototype, not an operational emergency system.
Risk scores are relative indices rather than calibrated probabilities.
Simulated depths are not hydraulic predictions, and route recommendations must
be verified by qualified emergency personnel and current local advisories.
