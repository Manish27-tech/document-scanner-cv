"""Batch evaluation and robustness sweeps on the synthetic corpus."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CONDITIONS, EVAL_SAMPLES, RANDOM_STATE
from .detect import detect_document
from .metrics import score_detection, summarize_scores
from .synth import generate_sample


def build_eval_set(n: int = EVAL_SAMPLES, seed: int = RANDOM_STATE) -> list[dict]:
    """Generate a balanced synthetic evaluation set across conditions."""
    samples = []
    for index in range(n):
        condition = CONDITIONS[index % len(CONDITIONS)]
        severity = float(np.round(0.5 + 0.5 * ((index // len(CONDITIONS)) % 3), 2))
        samples.append(
            generate_sample(
                sample_id=seed * 1000 + index,
                condition=condition,
                severity=severity,
                seed=seed * 1000 + index,
            )
        )
    return samples


def evaluate_samples(
    samples: list[dict],
    max_dimension: int | None = None,
    return_details: bool = False,
):
    """Run the detector on every sample and return per-image rows.

    With ``return_details=True`` the frame is accompanied by a list of dicts
    holding the original sample, detection result and score, which the figures
    and failure analysis use.
    """
    rows = []
    details = []
    for sample in samples:
        result = (
            detect_document(sample["image"])
            if max_dimension is None
            else detect_document(sample["image"], max_dimension=max_dimension)
        )
        score = score_detection(
            result["corners"],
            sample["corners"],
            sample["image"].shape[:2],
        )
        rows.append(
            {
                "sample_id": int(sample["sample_id"]),
                "condition": sample["condition"],
                "severity": float(sample["severity"]),
                "method": result["method"],
                "corner_error": score["corner_error"],
                "iou": score["iou"],
                "success": score["success"],
                "success_corner": score["success_corner"],
                "success_iou": score["success_iou"],
                "area_ratio": float(result.get("area_ratio", 0.0)),
                "runtime_ms": float(result.get("runtime_ms", 0.0)),
            }
        )
        if return_details:
            details.append({"sample": sample, "result": result, "score": score})
    frame = pd.DataFrame(rows)
    if return_details:
        return frame, details
    return frame


def condition_summary(per_image: pd.DataFrame) -> pd.DataFrame:
    """Aggregate success, error, IoU and runtime by condition."""
    grouped = per_image.groupby("condition", as_index=False).agg(
        samples=("sample_id", "count"),
        success_rate=("success", "mean"),
        corner_success_rate=("success_corner", "mean"),
        iou_success_rate=("success_iou", "mean"),
        median_corner_error=("corner_error", "median"),
        mean_iou=("iou", "mean"),
        mean_runtime_ms=("runtime_ms", "mean"),
    )
    order = {name: index for index, name in enumerate(CONDITIONS)}
    grouped["order"] = grouped["condition"].map(order)
    grouped = grouped.sort_values("order").drop(columns="order").reset_index(drop=True)
    for column in ("success_rate", "corner_success_rate", "iou_success_rate", "mean_iou"):
        grouped[column] = grouped[column].round(4)
    for column in ("median_corner_error", "mean_runtime_ms"):
        grouped[column] = grouped[column].round(3)
    return grouped


def robustness_sweep(repetitions: int = 8, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Measure success rate as each controllable degradation increases."""
    rows: list[dict] = []

    def append(axis: str, level: float, condition: str, repetition: int) -> None:
        sample_id = int(seed * 100000 + len(rows) * 17 + repetition)
        kwargs = {"angle_override": float(level)} if axis == "rotation_deg" else {}
        sample = generate_sample(
            sample_id=sample_id,
            condition=condition,
            severity=float(level) if axis != "rotation_deg" else 1.0,
            seed=sample_id,
            **kwargs,
        )
        result = detect_document(sample["image"])
        score = score_detection(result["corners"], sample["corners"], sample["image"].shape[:2])
        rows.append(
            {
                "axis": axis,
                "level": float(level),
                "condition": condition,
                "samples": 1,
                "success_rate": float(score["success"]),
            }
        )

    for angle in (0, 5, 10, 15, 20, 25):
        for repeat in range(repetitions):
            append("rotation_deg", float(angle), "rotate", repeat)

    for axis, condition, levels in (
        ("blur_sigma", "blur", (0.0, 0.5, 1.0, 1.5, 2.0)),
        ("noise_sigma", "noise", (0.0, 0.5, 1.0, 1.5, 2.0)),
        ("shadow_strength", "shadow", (0.0, 0.5, 1.0, 1.5)),
    ):
        for level in levels:
            for repeat in range(repetitions):
                append(axis, float(level), condition, repeat)

    frame = pd.DataFrame(rows)
    return (
        frame.groupby(["axis", "level", "condition"], as_index=False)
        .agg(samples=("success_rate", "count"), success_rate=("success_rate", "mean"))
        .assign(success_rate=lambda value: value["success_rate"].round(4))
    )


def aggregate_results(per_image: pd.DataFrame) -> dict:
    """Headline metrics for reports and the dashboard."""
    scores = [
        {
            "corner_error": row.corner_error,
            "iou": row.iou,
            "success_corner": bool(row.success_corner),
            "success_iou": bool(row.success_iou),
            "success": bool(row.success),
        }
        for row in per_image.itertuples(index=False)
    ]
    summary = summarize_scores(scores)
    summary["mean_runtime_ms"] = round(float(per_image["runtime_ms"].mean()), 3)
    summary["p95_runtime_ms"] = round(float(per_image["runtime_ms"].quantile(0.95)), 3)
    summary["methods"] = per_image["method"].value_counts().to_dict()
    return summary
