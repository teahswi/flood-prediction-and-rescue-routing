"""
Evaluation metrics for flood prediction quality.

Metrics implemented:
- Binary classification: Precision, Recall, F1, PR-AUC
- Probabilistic: Brier score, calibration
- Spatial extent: Intersection over Union (IoU), Dice coefficient
- Routing comparison: distance, time, hazard exposure

NOTE: Without actual ground-truth flood observations, these metrics
cannot be computed for real predictions. Use with synthetic/historical data.
"""
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass


def _precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple:
    """Compute a binary precision-recall curve without an optional ML runtime."""
    order = np.argsort(-y_score, kind="mergesort")
    scores = y_score[order]
    truth = y_true[order].astype(bool)
    distinct = np.r_[np.where(np.diff(scores))[0], len(scores) - 1]
    tp = np.cumsum(truth)[distinct]
    fp = np.cumsum(~truth)[distinct]
    positives = max(int(np.sum(truth)), 1)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    return (
        np.r_[precision, 1.0],
        np.r_[recall, 0.0],
    )


def _binary_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Compute trapezoidal area under a curve."""
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    return float(np.sum(np.diff(x_sorted) * (y_sorted[1:] + y_sorted[:-1]) / 2.0))


@dataclass
class ClassificationMetrics:
    """Holds binary classification evaluation results."""
    precision: float
    recall: float
    f1: float
    pr_auc: float
    brier_score: float
    threshold_used: float


@dataclass
class SpatialMetrics:
    """Holds spatial overlap evaluation results between predicted and observed masks."""
    iou: float                  # Intersection over Union
    dice: float                 # Dice coefficient (F1 of spatial overlap)
    precision_spatial: float    # TP / (TP + FP) in pixel space
    recall_spatial: float       # TP / (TP + FN) in pixel space
    total_predicted_km2: float  # Area of predicted flood extent
    total_observed_km2: float   # Area of observed flood extent


class FloodPredictionEvaluator:
    """
    Evaluator for flood prediction models.  Supports classification metrics,
    spatial overlap metrics, calibration analysis, and report generation.
    """

    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5,
    ) -> ClassificationMetrics:
        """
        Evaluate binary classification performance.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth binary labels (0 or 1).
        y_prob : np.ndarray
            Predicted flood probability / risk scores in [0, 1].
        threshold : float
            Decision threshold applied to y_prob to produce hard predictions.

        Returns
        -------
        ClassificationMetrics
        """
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)
        if y_true.shape != y_prob.shape or y_true.ndim != 1:
            raise ValueError("y_true and y_prob must be one-dimensional arrays of equal length")
        if not np.all(np.isin(y_true, (0.0, 1.0))):
            raise ValueError("y_true must contain only binary values 0 and 1")
        if not np.all((0.0 <= y_prob) & (y_prob <= 1.0)):
            raise ValueError("y_prob values must be in the range [0, 1]")
        y_pred = (y_prob >= threshold).astype(float)

        # --- precision, recall, f1 ---
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # --- PR-AUC ---
        prec_curve, rec_curve = _precision_recall_curve(y_true, y_prob)
        pr_auc = _binary_auc(rec_curve, prec_curve)

        # --- Brier score ---
        brier = float(np.mean((y_prob - y_true) ** 2))

        return ClassificationMetrics(
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            pr_auc=float(pr_auc),
            brier_score=float(brier),
            threshold_used=float(threshold),
        )

    def evaluate_spatial_extent(
        self,
        predicted_mask: np.ndarray,
        observed_mask: np.ndarray,
        cell_area_km2: float = 0.0009,
    ) -> SpatialMetrics:
        """
        Evaluate spatial overlap between a predicted flood mask and an
        observed flood mask.

        Parameters
        ----------
        predicted_mask : np.ndarray
            Boolean (or binary integer) 2-D array — True where flood is predicted.
        observed_mask : np.ndarray
            Boolean (or binary integer) 2-D array — True where flood was observed.
        cell_area_km2 : float
            Area of each raster cell in km².  Default 0.0009 km² ≈ 30 m × 30 m.

        Returns
        -------
        SpatialMetrics
        """
        pred = np.asarray(predicted_mask, dtype=bool)
        obs = np.asarray(observed_mask, dtype=bool)

        if pred.shape != obs.shape:
            raise ValueError(
                f"Mask shapes do not match: predicted {pred.shape} vs observed {obs.shape}"
            )

        intersection = np.sum(pred & obs)
        union = np.sum(pred | obs)
        sum_pred = np.sum(pred)
        sum_obs = np.sum(obs)

        iou = float(intersection / union) if union > 0 else 0.0
        dice = (
            float(2 * intersection / (sum_pred + sum_obs))
            if (sum_pred + sum_obs) > 0
            else 0.0
        )

        # Spatial precision and recall (pixel-level)
        precision_spatial = float(intersection / sum_pred) if sum_pred > 0 else 0.0
        recall_spatial = float(intersection / sum_obs) if sum_obs > 0 else 0.0

        total_predicted_km2 = float(sum_pred * cell_area_km2)
        total_observed_km2 = float(sum_obs * cell_area_km2)

        return SpatialMetrics(
            iou=iou,
            dice=dice,
            precision_spatial=precision_spatial,
            recall_spatial=recall_spatial,
            total_predicted_km2=total_predicted_km2,
            total_observed_km2=total_observed_km2,
        )

    def calibration_summary(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """
        Compute a reliability / calibration summary for flood risk scores.

        Returns a DataFrame with columns:
            bin_lower, bin_upper, mean_predicted, fraction_positive, count

        A well-calibrated model should have mean_predicted ≈ fraction_positive
        in each bin.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth binary labels.
        y_prob : np.ndarray
            Predicted probabilities in [0, 1].
        n_bins : int
            Number of equal-width probability bins.

        Returns
        -------
        pd.DataFrame
        """
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        records = []

        for i in range(n_bins):
            lo = bin_edges[i]
            hi = bin_edges[i + 1]
            # Include the upper edge in the last bin
            if i < n_bins - 1:
                mask = (y_prob >= lo) & (y_prob < hi)
            else:
                mask = (y_prob >= lo) & (y_prob <= hi)

            count = int(np.sum(mask))
            mean_predicted = float(np.mean(y_prob[mask])) if count > 0 else float((lo + hi) / 2)
            fraction_positive = float(np.mean(y_true[mask])) if count > 0 else 0.0

            records.append(
                {
                    "bin_lower": round(lo, 4),
                    "bin_upper": round(hi, 4),
                    "mean_predicted": round(mean_predicted, 4),
                    "fraction_positive": round(fraction_positive, 4),
                    "count": count,
                }
            )

        return pd.DataFrame(records)

    def generate_evaluation_report(
        self,
        classification: ClassificationMetrics,
        spatial: Optional[SpatialMetrics] = None,
    ) -> str:
        """
        Generate a formatted plain-text evaluation report.

        Parameters
        ----------
        classification : ClassificationMetrics
            Result from evaluate_classification().
        spatial : SpatialMetrics, optional
            Result from evaluate_spatial_extent().

        Returns
        -------
        str
            Human-readable multi-line report string.
        """
        lines = [
            "=" * 60,
            "  FLOOD PREDICTION EVALUATION REPORT",
            "=" * 60,
            "",
            "--- Classification Metrics ---",
            f"  Threshold Used   : {classification.threshold_used:.2f}",
            f"  Precision        : {classification.precision:.4f}",
            f"  Recall           : {classification.recall:.4f}",
            f"  F1 Score         : {classification.f1:.4f}",
            f"  PR-AUC           : {classification.pr_auc:.4f}",
            f"  Brier Score      : {classification.brier_score:.4f}  (lower is better)",
            "",
        ]

        if spatial is not None:
            lines += [
                "--- Spatial Extent Metrics ---",
                f"  IoU (Jaccard)    : {spatial.iou:.4f}",
                f"  Dice Coefficient : {spatial.dice:.4f}",
                f"  Spatial Precision: {spatial.precision_spatial:.4f}",
                f"  Spatial Recall   : {spatial.recall_spatial:.4f}",
                f"  Predicted Area   : {spatial.total_predicted_km2:.2f} km²",
                f"  Observed Area    : {spatial.total_observed_km2:.2f} km²",
                "",
            ]

        lines += [
            "--- Interpretation Guidance ---",
            "  NOTE: Metrics computed against synthetic / historical ground truth.",
            "  Real-world validation requires observed flood extent data.",
            "=" * 60,
        ]

        return "\n".join(lines)
