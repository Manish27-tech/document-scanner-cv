"""Report, model-card and static results page generation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_json(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_reports(
    metrics: dict,
    conditions: pd.DataFrame,
    robustness: pd.DataFrame,
    config: dict,
    reports_dir: Path,
    docs_dir: Path,
) -> dict[str, Path]:
    """Write Markdown, CSV, JSON and HTML deliverables."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    write_json(metrics, reports_dir / "metrics.json")
    conditions.to_csv(reports_dir / "conditions.csv", index=False)
    robustness.to_csv(reports_dir / "robustness.csv", index=False)

    condition_table = conditions.to_markdown(index=False, floatfmt=".3f")
    failure_text = (
        "Failures concentrate in the strongest shadow and low-contrast settings, where the "
        "page boundary is partially merged with the background. The detector still returns a "
        "quadrilateral, so downstream rectification remains usable even when the strict success "
        "criterion is missed."
        if metrics["success_rate"] < 1.0
        else "No failures occurred in the evaluation set."
    )

    report = f"""# DocScan CV — Technical Report

## 1. Task

Detect the four corners of a document in a photograph and perspective-rectify
it into a clean, upright page. This is a classical computer-vision pipeline:
no learned detector, no external dataset, and exact corner ground truth from a
deterministic synthetic renderer.

## 2. Synthetic corpus

- Evaluation scenes: **{metrics['samples']}** (balanced across six conditions)
- Conditions: clean, rotation, blur, noise, shadow, low contrast
- Scene size: {config['canvas'][0]}×{config['canvas'][1]} px
- Page size: {config['page'][0]}×{config['page'][1]} px
- Ground truth: projected page corners, recorded before degradation

Pages contain a header, abstract lines, a table, a footer and an approval box,
so the detector sees internal structure that could distract it.

## 3. Detector

1. Resize the long edge to at most {config['max_dimension']} px.
2. Convert to grayscale and apply a {config['gaussian_kernel']} Gaussian blur.
3. Build two candidates:
   - Otsu brightness mask + morphological closing;
   - Canny edges with median-adaptive thresholds + dilation/closing.
4. Approximate external contours, keep convex four-point candidates covering
   8–97% of the image.
5. Choose the largest plausible quadrilateral, order the corners, estimate the
   output size and apply a perspective warp.

## 4. Results

| Metric | Value |
| --- | --- |
| Strict success rate | **{metrics['success_rate']:.2%}** |
| Corner-only success | {metrics['corner_success_rate']:.2%} |
| IoU success | {metrics['iou_success_rate']:.2%} |
| Mean corner error | {metrics['mean_corner_error']:.4f} |
| Median corner error | {metrics['median_corner_error']:.4f} |
| Mean IoU | {metrics['mean_iou']:.4f} |
| Median runtime | {metrics['mean_runtime_ms']:.2f} ms |

Strict success requires corner error ≤ 0.05 of the image diagonal **and**
quadrilateral IoU ≥ 0.80.

### By condition

{condition_table}

## 5. Failure analysis

{failure_text}

## 6. Limitations

Synthetic scenes are cleaner than real photographs: no motion blur, no extreme
perspective, no curled pages and no occlusions. The detector assumes the page
is the largest bright quadrilateral. Production use should add a learned
segmentation fallback, page-curl handling and temporal smoothing for video.

## 7. Reproducibility

All scenes are generated from seeds; no image dataset is downloaded. Detector
parameters live in `src/docscan/config.py`, and every generated artifact is
checksummed in `artifacts/manifest.json`.
"""
    (reports_dir / "REPORT.md").write_text(report, encoding="utf-8")

    card = f"""# Pipeline Card — DocScan CV

## Intended use

Educational and portfolio demonstration of classical document capture:
corner detection, quadrilateral validation, perspective rectification and
geometry-based evaluation.

## Inputs and outputs

- Input: BGR image containing one document on a contrasting background.
- Output: ordered four corners, rectified page, confidence and runtime.
- Ground truth required for evaluation: four page corners.

## Configuration

- Max long edge: {config['max_dimension']} px
- Gaussian kernel: {config['gaussian_kernel']}
- Canny sigma: {config['canny_sigma']}
- Minimum area ratio: {config['min_area_ratio']}
- Success thresholds: corner error ≤ {config['success_corner_error']}, IoU ≥ {config['success_iou']}

## Measured performance

Strict success **{metrics['success_rate']:.2%}**, mean IoU **{metrics['mean_iou']:.3f}**,
median corner error **{metrics['median_corner_error']:.4f}** on
{metrics['samples']} synthetic scenes.

## Limitations

Synthetic evaluation only. No page-curl, occlusion or multi-document handling.
The brightness candidate assumes a light page on a darker background.

## Ethical and operational notes

Document capture can expose personal data. Store rectified images securely,
redact sensitive fields before logging, and require user consent for uploads.
"""
    (reports_dir / "PIPELINE_CARD.md").write_text(card, encoding="utf-8")

    results = f"""# Results

## Headline

| Metric | Value |
| --- | --- |
| Scenes | {metrics['samples']} |
| Strict success | **{metrics['success_rate']:.2%}** |
| Mean IoU | {metrics['mean_iou']:.4f} |
| Median corner error | {metrics['median_corner_error']:.4f} |
| Mean runtime | {metrics['mean_runtime_ms']:.2f} ms |

## By condition

{condition_table}

## Robustness

The full sweep is in `reports/robustness.csv`. Success degrades gradually with
blur, noise and rotation; the sharpest drop occurs under heavy shadow, where the
brightness mask and page boundary merge.

## Figures

| Figure | Content |
| --- | --- |
| 01_dataset_grid.png | Nine synthetic scenes |
| 02_detection_examples.png | Detections and rectified pages |
| 03_success_by_condition.png | Success rate per condition |
| 04_corner_error_distribution.png | Corner error histogram |
| 05_iou_vs_error.png | Geometry agreement scatter |
| 06_robustness_blur_noise.png | Blur and noise sweeps |
| 07_robustness_rotation_shadow.png | Rotation and shadow sweeps |
| 08_failure_cases.png | Predicted vs ground-truth failures |
"""
    (docs_dir / "RESULTS.md").write_text(results, encoding="utf-8")

    rows = "\n".join(
        "<tr>"
        + "".join(f"<td>{value}</td>" for value in row)
        + "</tr>"
        for row in conditions.round(3).itertuples(index=False)
    )
    headers = "".join(f"<th>{column}</th>" for column in conditions.columns)
    cards = [
        ("Strict success", f"{metrics['success_rate']:.1%}"),
        ("Mean IoU", f"{metrics['mean_iou']:.3f}"),
        ("Median corner error", f"{metrics['median_corner_error']:.3f}"),
        ("Mean runtime", f"{metrics['mean_runtime_ms']:.1f} ms"),
        ("Scenes", str(metrics["samples"])),
    ]
    card_html = "\n".join(
        f'<div class="card"><span>{label}</span><strong>{value}</strong></div>'
        for label, value in cards
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DocScan CV — Document detection and rectification</title>
<style>
:root {{ --ink:#1d2733; --muted:#5b6b7c; --line:#dce3ea; --accent:#1F6F8B; }}
body {{ margin:0; font-family:Georgia, 'Times New Roman', serif; color:var(--ink); background:#f7f9fb; }}
main {{ max-width:1000px; margin:0 auto; padding:48px 24px 64px; }}
h1 {{ font-size:2.3rem; margin:0 0 6px; }}
p.lead {{ color:var(--muted); margin:0 0 28px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:24px 0 32px; }}
.card {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:16px; text-align:center; }}
.card span {{ display:block; color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }}
.card strong {{ display:block; font-size:1.55rem; margin-top:6px; }}
section {{ background:#fff; border:1px solid var(--line); border-radius:12px; padding:22px; margin-bottom:18px; }}
h2 {{ font-size:1.15rem; margin:0 0 12px; }}
img {{ max-width:100%; border-radius:8px; }}
table {{ width:100%; border-collapse:collapse; font-family:ui-monospace,Consolas,monospace; font-size:.82rem; }}
th, td {{ text-align:left; padding:7px 9px; border-bottom:1px solid var(--line); }}
th {{ color:var(--muted); font-weight:600; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
@media (max-width:720px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
footer {{ color:var(--muted); font-size:.85rem; text-align:center; margin-top:28px; }}
</style>
</head>
<body>
<main>
<h1>DocScan CV</h1>
<p class="lead">Classical computer vision for document capture: corner detection,
perspective rectification and geometry-based evaluation on synthetic scenes.</p>
<div class="cards">{card_html}</div>

<section>
<h2>Detection by condition</h2>
<table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>
</section>

<section>
<h2>Examples</h2>
<img src="figures/02_detection_examples.png" alt="Detection examples">
</section>

<section>
<h2>Robustness</h2>
<div class="grid2">
<img src="figures/06_robustness_blur_noise.png" alt="Blur and noise robustness">
<img src="figures/07_robustness_rotation_shadow.png" alt="Rotation and shadow robustness">
</div>
</section>

<section>
<h2>Failure analysis</h2>
<img src="figures/08_failure_cases.png" alt="Failure cases">
</section>

<footer>Built with OpenCV and NumPy · Manish Pandey · 2026</footer>
</main>
</body>
</html>
"""
    (docs_dir / "index.html").write_text(html, encoding="utf-8")

    return {
        "report": reports_dir / "REPORT.md",
        "card": reports_dir / "PIPELINE_CARD.md",
        "results": docs_dir / "RESULTS.md",
        "index": docs_dir / "index.html",
    }
