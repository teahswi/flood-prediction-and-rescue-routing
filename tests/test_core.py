import unittest

import networkx as nx
import numpy as np

from src.evaluation.flood_metrics import FloodPredictionEvaluator
from src.flood.features import (
    FloodFeatureVector,
    HistoricalFeatures,
    RainfallFeatures,
    TerrainFeatures,
)
from src.flood.risk_model import FloodRiskModel
from src.routing.graph_builder import FloodAwareGraphBuilder
from src.routing.hazard_cost import HazardCostFunction


class FloodEngineTests(unittest.TestCase):
    def test_risk_marks_missing_historical_and_drainage_inputs(self):
        features = FloodFeatureVector(
            lat=19.0,
            lon=72.0,
            timestamp=__import__("datetime").datetime.utcnow(),
            rainfall=RainfallFeatures(40, 80, 120, 5, 15, 25, "Orange"),
            terrain=TerrainFeatures(2, 0, True, 20),
            historical=HistoricalFeatures(),
            data_quality={"rainfall": True, "terrain": True},
        )
        assessment = FloodRiskModel().predict(features)
        self.assertIn("historical", assessment.data_gaps)
        self.assertIn("drainage", assessment.data_gaps)
        self.assertGreaterEqual(assessment.composite_risk_score, 0.0)
        self.assertLessEqual(assessment.composite_risk_score, 1.0)

    def test_evaluation_metrics_reject_invalid_probabilities(self):
        evaluator = FloodPredictionEvaluator()
        with self.assertRaises(ValueError):
            evaluator.evaluate_classification(np.array([0, 1]), np.array([0.2, 1.2]))


class RoutingTests(unittest.TestCase):
    def test_blocked_edges_are_removed_and_alternative_path_remains(self):
        graph = nx.MultiDiGraph()
        graph.add_node(1, x=72.0, y=19.0)
        graph.add_node(2, x=72.01, y=19.0)
        graph.add_node(3, x=72.0, y=19.01)
        graph.add_edge(1, 2, length=100, travel_time=10)
        graph.add_edge(2, 3, length=100, travel_time=10)
        graph.add_edge(1, 3, length=300, travel_time=30)

        result = FloodAwareGraphBuilder(HazardCostFunction()).build(
            graph,
            flood_risk_fn=lambda lat, lon: 0.9 if abs(lon - 72.005) < 0.003 else 0.1,
        )

        self.assertEqual(result.num_blocked_edges, 2)
        self.assertEqual(result.graph_feasible.number_of_edges(), 1)
        self.assertTrue(nx.has_path(result.graph_feasible, 1, 3))


if __name__ == "__main__":
    unittest.main()
