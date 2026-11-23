"""Matplotlib figures for dataset, detection, robustness and failures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .detect import draw_detection  # noqa: E402
from .geometry import order_corners  # noqa: E402

INK = "#243447"
ACCENT = "#1F6F8B"
GOOD = "#2A9D8F"
BAD = "#D1495B"
GRID = {"alpha": 0.25, "linewidth": 0.6}


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def dataset_grid(samples: list[dict], path: Path) -> Path:
    fig, axes = plt.subplots(3, 3, figsize=(10, 7.5))
    for ax, sample in zip(axes.ravel(), samples[:9], strict=False):
        ax.imshow(cv2_to_rgb(sample["image"]))
        ax.set_title(sample["condition"], fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[len(samples[:9]) :]:
        ax.axis("off")
    fig.suptitle("Synthetic document scenes with exact corner labels", fontsize=12)
    return _save(fig, path)


def cv2_to_rgb(image: np.ndarray) -> np.ndarray:
    return image[:, :, ::-1]


def detection_examples(details: list[dict], path: Path) -> Path:
    selected = [item for item in details if item["score"]["success"]][:4]
    if not selected:
        selected = details[:4]
    fig, axes = plt.subplots(len(selected), 2, figsize=(10, 3 * len(selected)))
    for row, item in enumerate(selected):
        image = item["sample"]["image"]
        truth = item["sample"]["corners"]
        predicted = item["result"]["corners"]
        overlay = draw_detection(image, predicted, colour=(45, 200, 130))
        overlay = overlay.copy()
        import cv2

        cv2.polylines(
            overlay,
            [np.round(order_corners(truth)).astype(np.int32)],
            True,
            (60, 120, 240),
            2,
            cv2.LINE_AA,
        )
        axes[row, 0].imshow(cv2_to_rgb(overlay))
        axes[row, 0].set_title(f"Detection · {item['sample']['condition']}", fontsize=9)
        axes[row, 0].axis("off")
        warped = item["result"]["warped"]
        if warped is not None:
            axes[row, 1].imshow(cv2_to_rgb(warped))
        axes[row, 1].set_title("Rectified page", fontsize=9)
        axes[row, 1].axis("off")
    fig.suptitle("Detected quadrilateral (green) and ground truth (blue)", fontsize=12)
    return _save(fig, path)


def success_by_condition(conditions: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(conditions["condition"], conditions["success_rate"], color=ACCENT, width=0.6)
    for bar, value in zip(bars, conditions["success_rate"], strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.0%}", ha="center")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Strict success rate")
    ax.set_title("Detection success by condition")
    ax.grid(axis="y", **GRID)
    return _save(fig, path)


def corner_error_distribution(per_image: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(per_image["corner_error"], bins=18, color=ACCENT, alpha=0.85)
    ax.axvline(0.05, color=BAD, linestyle="--", linewidth=1.2, label="Success threshold (5%)")
    ax.set_xlabel("Mean corner error / image diagonal")
    ax.set_ylabel("Samples")
    ax.set_title("Corner error distribution")
    ax.legend(frameon=False)
    ax.grid(axis="y", **GRID)
    return _save(fig, path)


def iou_vs_error(per_image: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    colours = np.where(per_image["success"], GOOD, BAD)
    ax.scatter(per_image["corner_error"], per_image["iou"], c=colours, alpha=0.75, edgecolor="white")
    ax.axvline(0.05, color=BAD, linestyle="--", linewidth=1)
    ax.axhline(0.80, color=BAD, linestyle="--", linewidth=1)
    ax.set_xlabel("Corner error")
    ax.set_ylabel("IoU")
    ax.set_title("Geometry agreement (green = success)")
    ax.grid(**GRID)
    return _save(fig, path)


def robustness_blur_noise(robustness: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, axis_name, title in (
        (axes[0], "blur_sigma", "Gaussian blur"),
        (axes[1], "noise_sigma", "Sensor noise"),
    ):
        subset = robustness[robustness["axis"] == axis_name]
        ax.plot(subset["level"], subset["success_rate"], marker="o", color=ACCENT)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Severity")
        ax.set_ylabel("Success rate")
        ax.set_title(title)
        ax.grid(**GRID)
    fig.suptitle("Robustness to blur and noise", fontsize=12)
    return _save(fig, path)


def robustness_rotation_shadow(robustness: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, axis_name, title in (
        (axes[0], "rotation_deg", "Page rotation"),
        (axes[1], "shadow_strength", "Shadow strength"),
    ):
        subset = robustness[robustness["axis"] == axis_name]
        ax.plot(subset["level"], subset["success_rate"], marker="o", color=ACCENT)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Degrees" if axis_name == "rotation_deg" else "Severity")
        ax.set_ylabel("Success rate")
        ax.set_title(title)
        ax.grid(**GRID)
    fig.suptitle("Robustness to rotation and shadow", fontsize=12)
    return _save(fig, path)


def failure_cases(details: list[dict], path: Path) -> Path:
    failures = [item for item in details if not item["score"]["success"]][:4]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    if not failures:
        for ax in axes:
            ax.axis("off")
        axes[0].text(0.5, 0.5, "No failures in this evaluation set", ha="center", va="center")
    for ax, item in zip(axes, failures, strict=False):
        image = item["sample"]["image"]
        truth = item["sample"]["corners"]
        predicted = item["result"]["corners"]
        overlay = draw_detection(image, predicted, colour=(60, 70, 220))
        import cv2

        cv2.polylines(
            overlay,
            [np.round(order_corners(truth)).astype(np.int32)],
            True,
            (60, 190, 120),
            2,
            cv2.LINE_AA,
        )
        ax.imshow(cv2_to_rgb(overlay))
        score = item["score"]
        ax.set_title(
            f"{item['sample']['condition']} · err={score['corner_error']:.3f} · IoU={score['iou']:.2f}",
            fontsize=9,
        )
        ax.axis("off")
    for ax in axes[len(failures) :]:
        ax.axis("off")
    fig.suptitle("Failure cases — predicted (blue) vs ground truth (green)", fontsize=12)
    return _save(fig, path)


def make_all(
    samples: list[dict],
    details: list[dict],
    per_image: pd.DataFrame,
    conditions: pd.DataFrame,
    robustness: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Path]:
    """Render every project figure and return their paths."""
    return {
        "01_dataset_grid": dataset_grid(samples, out_dir / "01_dataset_grid.png"),
        "02_detection_examples": detection_examples(details, out_dir / "02_detection_examples.png"),
        "03_success_by_condition": success_by_condition(
            conditions, out_dir / "03_success_by_condition.png"
        ),
        "04_corner_error_distribution": corner_error_distribution(
            per_image, out_dir / "04_corner_error_distribution.png"
        ),
        "05_iou_vs_error": iou_vs_error(per_image, out_dir / "05_iou_vs_error.png"),
        "06_robustness_blur_noise": robustness_blur_noise(
            robustness, out_dir / "06_robustness_blur_noise.png"
        ),
        "07_robustness_rotation_shadow": robustness_rotation_shadow(
            robustness, out_dir / "07_robustness_rotation_shadow.png"
        ),
        "08_failure_cases": failure_cases(details, out_dir / "08_failure_cases.png"),
    }
