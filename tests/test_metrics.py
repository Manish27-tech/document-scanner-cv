import numpy as np

from docscan.metrics import score_detection, summarize_scores

TRUTH = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype="float32")


def test_perfect_detection_scores_one():
    score = score_detection(TRUTH, TRUTH, (100, 100))
    assert score["corner_error"] == 0.0
    assert score["iou"] == 1.0
    assert score["success"] is True


def test_missing_detection_fails():
    score = score_detection(None, TRUTH, (100, 100))
    assert score["corner_error"] == 1.0
    assert score["iou"] == 0.0
    assert score["success"] is False


def test_shifted_detection_may_fail():
    shifted = TRUTH + np.array([30, 30], dtype="float32")
    score = score_detection(shifted, TRUTH, (100, 100))
    assert score["success"] is False
    assert 0.0 < score["corner_error"] < 1.0


def test_summary_values():
    scores = [
        score_detection(TRUTH, TRUTH, (100, 100)),
        score_detection(None, TRUTH, (100, 100)),
    ]
    summary = summarize_scores(scores)
    assert summary["samples"] == 2
    assert summary["success_rate"] == 0.5
    assert 0.0 <= summary["mean_iou"] <= 1.0
