import numpy as np

from docscan.detect import detect_document, draw_detection
from docscan.synth import generate_sample


def test_detects_clean_document():
    sample = generate_sample(sample_id=21, condition="clean", seed=21, canvas_size=(800, 620))
    result = detect_document(sample["image"])
    assert result["corners"] is not None
    assert result["corners"].shape == (4, 2)
    assert result["warped"] is not None
    assert result["method"] in {"brightness", "edges"}
    assert result["runtime_ms"] > 0


def test_warped_page_is_portrait():
    sample = generate_sample(sample_id=22, condition="rotate", seed=22, canvas_size=(800, 620))
    result = detect_document(sample["image"])
    height, width = result["warped"].shape[:2]
    assert height > width


def test_draw_detection_returns_copy():
    sample = generate_sample(sample_id=23, condition="clean", seed=23, canvas_size=(600, 480))
    result = detect_document(sample["image"])
    canvas = draw_detection(sample["image"], result["corners"])
    assert canvas.shape == sample["image"].shape
    assert not np.array_equal(canvas, sample["image"])


def test_empty_image_raises():
    import pytest

    with pytest.raises(ValueError):
        detect_document(np.array([]))
