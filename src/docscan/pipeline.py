"""End-to-end orchestration for DocScan CV."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2

from .config import (
    ARTIFACTS,
    CANNY_SIGMA,
    CANVAS_SIZE,
    CONDITIONS,
    DOCS_DIR,
    EVAL_SAMPLES,
    GAUSSIAN_KERNEL,
    MAX_AREA_RATIO,
    MAX_DIMENSION,
    MIN_AREA_RATIO,
    PAGE_SIZE,
    RANDOM_STATE,
    REPORTS_DIR,
    ROOT,
    SUCCESS_CORNER_ERROR,
    SUCCESS_IOU,
)
from .detect import detect_document, draw_detection
from .evaluate import (
    aggregate_results,
    build_eval_set,
    condition_summary,
    evaluate_samples,
    robustness_sweep,
)
from .manifest import build_manifest
from .plots import make_all
from .report import write_json, write_reports


def _copy_figures(figures: dict[str, Path], targets: list[Path]) -> None:
    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
        for source in figures.values():
            shutil.copy2(source, target / source.name)


def _save_examples(details: list[dict], directory: Path, limit: int = 2) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for item in details[:limit]:
        sample = item["sample"]
        result = item["result"]
        identifier = sample["sample_id"]
        overlay = draw_detection(sample["image"], result["corners"])
        overlay_path = directory / f"{identifier}_overlay.png"
        cv2.imwrite(str(overlay_path), overlay)
        paths.append(overlay_path)
        if result["warped"] is not None:
            warped_path = directory / f"{identifier}_rectified.png"
            cv2.imwrite(str(warped_path), result["warped"])
            paths.append(warped_path)
    return paths


def run_pipeline(
    n_samples: int = EVAL_SAMPLES,
    seed: int = RANDOM_STATE,
    robustness_reps: int = 8,
    *,
    artifacts_dir: Path = ARTIFACTS,
    reports_dir: Path = REPORTS_DIR,
    docs_dir: Path = DOCS_DIR,
    show_progress: bool = False,
) -> dict:
    """Generate the corpus, evaluate the detector and publish every artifact."""
    artifacts_dir = Path(artifacts_dir)
    reports_dir = Path(reports_dir)
    docs_dir = Path(docs_dir)
    figures_dir = artifacts_dir / "figures"
    dashboard_dir = artifacts_dir / "dashboard"
    for directory in (artifacts_dir, reports_dir, docs_dir, figures_dir, dashboard_dir):
        directory.mkdir(parents=True, exist_ok=True)

    samples = build_eval_set(n=n_samples, seed=seed)
    if show_progress:
        print(f"generated {len(samples)} scenes")
    per_image, details = evaluate_samples(samples, return_details=True)
    conditions = condition_summary(per_image)
    if show_progress:
        print("running robustness sweep")
    robustness = robustness_sweep(repetitions=robustness_reps, seed=seed)
    metrics = aggregate_results(per_image)

    config = {
        "canvas": list(CANVAS_SIZE),
        "page": list(PAGE_SIZE),
        "max_dimension": MAX_DIMENSION,
        "gaussian_kernel": list(GAUSSIAN_KERNEL),
        "canny_sigma": CANNY_SIGMA,
        "min_area_ratio": MIN_AREA_RATIO,
        "max_area_ratio": MAX_AREA_RATIO,
        "success_corner_error": SUCCESS_CORNER_ERROR,
        "success_iou": SUCCESS_IOU,
        "conditions": CONDITIONS,
        "seed": seed,
        "samples": len(samples),
    }

    figures = make_all(samples, details, per_image, conditions, robustness, figures_dir)
    _copy_figures(figures, [reports_dir / "figures", docs_dir / "figures"])
    examples = _save_examples(details, dashboard_dir / "examples")

    dataset_summary = {
        "samples": len(samples),
        "conditions": {name: int((per_image["condition"] == name).sum()) for name in CONDITIONS},
        "canvas": list(CANVAS_SIZE),
        "page": list(PAGE_SIZE),
        "success_threshold": SUCCESS_CORNER_ERROR,
        "iou_threshold": SUCCESS_IOU,
    }
    write_json(dataset_summary, dashboard_dir / "dataset_summary.json")
    write_json(metrics, dashboard_dir / "metrics.json")
    write_json(config, artifacts_dir / "scan_config.json")
    write_json(config, dashboard_dir / "scan_config.json")
    per_image.to_csv(dashboard_dir / "per_image.csv", index=False)
    conditions.to_csv(dashboard_dir / "conditions.csv", index=False)
    robustness.to_csv(dashboard_dir / "robustness.csv", index=False)

    deliverables = write_reports(metrics, conditions, robustness, config, reports_dir, docs_dir)

    manifest_files = [path for path in artifacts_dir.rglob("*") if path.is_file()]
    manifest_files += [path for path in reports_dir.rglob("*") if path.is_file()]
    manifest_files += [path for path in docs_dir.rglob("*") if path.is_file()]
    manifest = build_manifest(
        manifest_files,
        ROOT,
        output=artifacts_dir / "manifest.json",
    )

    return {
        "metrics": metrics,
        "conditions": conditions,
        "robustness": robustness,
        "per_image": per_image,
        "details": details,
        "figures": figures,
        "examples": examples,
        "manifest": manifest,
        "config": config,
        "deliverables": deliverables,
    }


def scan_file(path: str | Path) -> dict:
    """Detect and rectify a document in an image file."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return detect_document(image)


def scan_summary(result: dict) -> dict:
    """Small JSON-friendly summary of a single scan."""
    corners = result.get("corners")
    return {
        "method": result.get("method"),
        "confidence": round(float(result.get("confidence", 0.0)), 4),
        "area_ratio": round(float(result.get("area_ratio", 0.0)), 4),
        "runtime_ms": round(float(result.get("runtime_ms", 0.0)), 2),
        "corners": None if corners is None else [[round(float(x), 1), round(float(y), 1)] for x, y in corners],
        "rectified_size": None
        if result.get("warped") is None
        else [int(result["warped"].shape[1]), int(result["warped"].shape[0])],
    }
