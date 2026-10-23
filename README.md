# Flood Prediction & Rescue Routing

This project is a Streamlit prototype for flood-risk screening, simplified
flood-spread simulation, and hazard-aware rescue route comparison.

The prediction component is a deterministic risk index, not a trained or
validated forecasting model. It currently uses live weather observations,
short-range forecasts, and terrain screening. Historical flood labels and
drainage observations are not bundled, so those inputs are reported as
missing rather than claimed as validated evidence.

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
  `src.routing.route_planner.RoutePlanner` for hazard-aware routing. Routes
  use an OSM road graph, attach risk-derived edge costs, remove edges at the
  impassable threshold, and recalculate when the risk input changes.
- `src.evaluation.flood_metrics.FloodPredictionEvaluator` and
  `src.evaluation.route_metrics.compare_routes` for offline evaluation.

## Important limitations

This is a decision-support prototype, not an operational emergency system.
Risk scores are relative indices rather than calibrated probabilities.
Simulated depths are not hydraulic predictions, and route recommendations must
be verified by qualified emergency personnel and current local advisories.
Sample incidents and shelters in the dashboard are demonstration records.
They are not a live emergency feed.

## Tests

Run the dependency-light core checks with:

```text
python -m unittest discover -s tests -v
```

Evaluation against real flood observations is intentionally not reported until
an independent, time- or location-held-out labelled dataset is supplied.
