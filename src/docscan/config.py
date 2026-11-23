"""Central configuration for the DocScan CV pipeline."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
SAMPLES_LABELS = DATA_DIR / "samples.csv"
GENERATED_DIR = DATA_DIR / "generated"

ARTIFACTS = ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS / "figures"
DASHBOARD_DIR = ARTIFACTS / "dashboard"
SCAN_CONFIG = ARTIFACTS / "scan_config.json"
MANIFEST_PATH = ARTIFACTS / "manifest.json"

REPORTS_DIR = ROOT / "reports"
REPORT_FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = ROOT / "docs"
DOCS_FIGURES_DIR = DOCS_DIR / "figures"

RANDOM_STATE = 42
CANVAS_SIZE = (1100, 850)  # width, height
PAGE_SIZE = (700, 1000)  # width, height
EVAL_SAMPLES = 120
CONDITIONS = ["clean", "rotate", "blur", "noise", "shadow", "lowcontrast"]

# Detector defaults
MAX_DIMENSION = 1000
GAUSSIAN_KERNEL = (5, 5)
CANNY_SIGMA = 0.66
MIN_AREA_RATIO = 0.08
MAX_AREA_RATIO = 0.97
APPROX_EPSILON = 0.02
SUCCESS_CORNER_ERROR = 0.05  # fraction of image diagonal
SUCCESS_IOU = 0.80

LABELS = {"document": 1, "background": 0}
