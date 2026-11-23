"""Dashboard data loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DASHBOARD = ARTIFACTS / "dashboard"
FIGURES = ARTIFACTS / "figures"
SAMPLES = ROOT / "data" / "samples"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_all() -> dict:
    return {
        "metrics": _read_json(DASHBOARD / "metrics.json"),
        "summary": _read_json(DASHBOARD / "dataset_summary.json"),
        "config": _read_json(DASHBOARD / "scan_config.json"),
        "per_image": _read_csv(DASHBOARD / "per_image.csv"),
        "conditions": _read_csv(DASHBOARD / "conditions.csv"),
        "robustness": _read_csv(DASHBOARD / "robustness.csv"),
        "samples": _read_csv(ROOT / "data" / "samples.csv"),
        "figures": FIGURES,
        "samples_dir": SAMPLES,
        "examples": DASHBOARD / "examples",
    }
