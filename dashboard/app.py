"""Streamlit dashboard for DocScan CV."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dashboard"))

from data import load_all  # noqa: E402
from docscan.detect import detect_document, draw_detection  # noqa: E402

st.set_page_config(page_title="DocScan CV", page_icon="📄", layout="wide")
bundle = load_all()
metrics = bundle["metrics"]
figures = bundle["figures"]

st.title("DocScan CV — document detection and rectification")
st.caption("Classical computer vision: corner detection, quadrilateral validation and perspective warping.")

if not metrics:
    st.warning("No evaluation artifacts found. Run `python run.py evaluate`.")
    st.stop()

cards = st.columns(5)
for column, (label, value) in zip(
    cards,
    [
        ("Strict success", f"{metrics['success_rate']:.1%}"),
        ("Mean IoU", f"{metrics['mean_iou']:.3f}"),
        ("Median corner error", f"{metrics['median_corner_error']:.3f}"),
        ("Mean runtime", f"{metrics['mean_runtime_ms']:.1f} ms"),
        ("Scenes", str(metrics["samples"])),
    ],
    strict=True,
):
    column.metric(label, value)

overview, detector_tab, evaluation, robustness_tab = st.tabs(
    ["Overview", "Try the detector", "Evaluation", "Robustness"]
)

with overview:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Condition summary")
        st.dataframe(bundle["conditions"], hide_index=True, use_container_width=True)
    with right:
        st.subheader("Committed samples")
        samples = bundle["samples"]
        if not samples.empty:
            chosen = st.selectbox("Sample", samples["image"].tolist())
            image_path = bundle["samples_dir"] / chosen
            if image_path.exists():
                st.image(str(image_path), caption=chosen, use_container_width=True)
    st.image(str(figures / "01_dataset_grid.png"), use_container_width=True)

with detector_tab:
    st.write("Upload a document photo or use one of the committed synthetic samples.")
    upload = st.file_uploader("Image", type=["png", "jpg", "jpeg"])
    image = None
    if upload is not None:
        import numpy as np

        data = np.frombuffer(upload.read(), np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    else:
        samples = bundle["samples"]
        if not samples.empty:
            choice = st.selectbox("Or pick a sample", samples["image"].tolist(), key="detector_sample")
            path = bundle["samples_dir"] / choice
            if path.exists():
                image = cv2.imread(str(path))

    if image is not None:
        result = detect_document(image)
        overlay = draw_detection(image, result["corners"])
        left, right = st.columns(2)
        left.image(overlay[:, :, ::-1], caption="Detected quadrilateral", use_container_width=True)
        if result["warped"] is not None:
            right.image(result["warped"][:, :, ::-1], caption="Rectified page", use_container_width=True)
        else:
            right.warning("No quadrilateral found.")
        st.json(
            {
                "method": result["method"],
                "confidence": round(result["confidence"], 4),
                "area_ratio": round(result["area_ratio"], 4),
                "runtime_ms": round(result["runtime_ms"], 2),
            }
        )

with evaluation:
    a, b = st.columns(2)
    a.image(str(figures / "03_success_by_condition.png"), use_container_width=True)
    b.image(str(figures / "04_corner_error_distribution.png"), use_container_width=True)
    a.image(str(figures / "05_iou_vs_error.png"), use_container_width=True)
    b.image(str(figures / "02_detection_examples.png"), use_container_width=True)
    st.subheader("Per-image results")
    if not bundle["per_image"].empty:
        condition = st.selectbox("Filter condition", ["all", *bundle["per_image"]["condition"].unique()])
        view = bundle["per_image"]
        if condition != "all":
            view = view[view["condition"] == condition]
        st.dataframe(
            view.sort_values("corner_error", ascending=False),
            hide_index=True,
            use_container_width=True,
        )
    st.image(str(figures / "08_failure_cases.png"), use_container_width=True)

with robustness_tab:
    st.image(str(figures / "06_robustness_blur_noise.png"), use_container_width=True)
    st.image(str(figures / "07_robustness_rotation_shadow.png"), use_container_width=True)
    st.dataframe(bundle["robustness"], hide_index=True, use_container_width=True)
