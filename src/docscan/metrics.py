"""Detection metrics: corner error, IoU and success criteria."""

from __future__ import annotations

import numpy as np

from .config import SUCCESS_CORNER_ERROR, SUCCESS_IOU
from .geometry import corner_error, quad_iou


def score_detection(
    predicted: np.ndarray | None,
    truth: np.ndarray,
    shape: tuple[int, int],
    corner_threshold: float = SUCCESS_CORNER_ERROR,
    iou_threshold: float = SUCCESS_IOU,
) -> dict:
    """Score one detection against ground truth.

    ``success`` is strict: the mean corner error must be within the threshold
    *and* the quadrilateral IoU must clear its own threshold. The two
    components are also reported separately for diagnostics.
    """
    if predicted is None:
        return {
            "corner_error": 1.0,
            "iou": 0.0,
            "success_corner": False,
            "success_iou": False,
            "success": False,
        }
    error = corner_error(predicted, truth, shape)
    iou = quad_iou(predicted, truth, shape)
    success_corner = error <= corner_threshold
    success_iou = iou >= iou_threshold
    return {
        "corner_error": float(error),
        "iou": float(iou),
        "success_corner": bool(success_corner),
        "success_iou": bool(success_iou),
        "success": bool(success_corner and success_iou),
    }


def summarize_scores(scores: list[dict]) -> dict:
    """Aggregate a list of detection scores into report-friendly numbers."""
    if not scores:
        return {}
    errors = np.array([score["corner_error"] for score in scores])
    ious = np.array([score["iou"] for score in scores])
    successes = np.array([score["success"] for score in scores], dtype=float)
    return {
        "samples": int(len(scores)),
        "success_rate": round(float(successes.mean()), 4),
        "corner_success_rate": round(float(np.mean([s["success_corner"] for s in scores])), 4),
        "iou_success_rate": round(float(np.mean([s["success_iou"] for s in scores])), 4),
        "mean_corner_error": round(float(errors.mean()), 4),
        "median_corner_error": round(float(np.median(errors)), 4),
        "p95_corner_error": round(float(np.quantile(errors, 0.95)), 4),
        "mean_iou": round(float(ious.mean()), 4),
        "median_iou": round(float(np.median(ious)), 4),
        "min_iou": round(float(ious.min()), 4),
        "max_iou": round(float(ious.max()), 4),
    }
