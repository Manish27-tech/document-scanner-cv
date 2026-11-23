# Data Dictionary

## `data/samples/*.png`

Twelve committed synthetic scenes (two per condition).

## `data/samples.csv`

| Column | Type | Description |
| --- | --- | --- |
| image | string | File name under `data/samples/` |
| sample_id | integer | Deterministic scene identifier |
| condition | string | clean / rotate / blur / noise / shadow / lowcontrast |
| severity | float | Degradation severity |
| c0 … c7 | float | Ground-truth corners flattened as x1,y1,x2,y2,x3,y3,x4,y4 |

Corners are ordered top-left, top-right, bottom-right, bottom-left.

## Dashboard feed (`artifacts/dashboard/`)

| File | Description |
| --- | --- |
| `metrics.json` | Headline success, IoU, corner error and runtime metrics |
| `dataset_summary.json` | Scene count, condition counts, canvas and page size |
| `scan_config.json` | Detector parameters and thresholds |
| `per_image.csv` | Sample id, condition, method, error, IoU, success, runtime |
| `conditions.csv` | Aggregated success, error, IoU and runtime per condition |
| `robustness.csv` | Success rate per degradation axis and level |
| `examples/` | Overlay and rectified outputs for two successful scenes |

## Artifacts

| File | Description |
| --- | --- |
| `figures/01_dataset_grid.png` | Nine generated scenes |
| `figures/02_detection_examples.png` | Detection and rectification examples |
| `figures/03_success_by_condition.png` | Strict success per condition |
| `figures/04_corner_error_distribution.png` | Corner error histogram |
| `figures/05_iou_vs_error.png` | IoU versus corner error |
| `figures/06_robustness_blur_noise.png` | Blur and noise sweeps |
| `figures/07_robustness_rotation_shadow.png` | Rotation and shadow sweeps |
| `figures/08_failure_cases.png` | Predicted versus ground-truth failures |
| `scan_config.json` | Reproducible detector configuration |
| `manifest.json` | SHA-256 checksums for generated files |

## Reports

| File | Description |
| --- | --- |
| `reports/REPORT.md` | Technical report with method, results and limitations |
| `reports/PIPELINE_CARD.md` | Intended use, configuration and measured performance |
| `reports/metrics.json` | Machine-readable headline metrics |
| `reports/conditions.csv` | Per-condition results |
| `reports/robustness.csv` | Full robustness sweep |
