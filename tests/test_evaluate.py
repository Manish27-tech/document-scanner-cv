from docscan.evaluate import (
    aggregate_results,
    build_eval_set,
    condition_summary,
    evaluate_samples,
    robustness_sweep,
)


def test_eval_set_is_balanced():
    samples = build_eval_set(n=6)
    conditions = [sample["condition"] for sample in samples]
    assert len(conditions) == 6
    assert len(set(conditions)) == 6


def test_evaluate_samples_returns_rows_and_details():
    samples = build_eval_set(n=6)
    frame, details = evaluate_samples(samples, return_details=True)
    assert len(frame) == 6
    assert len(details) == 6
    assert {"corner_error", "iou", "success", "runtime_ms"} <= set(frame.columns)
    assert frame["success"].dtype == bool


def test_condition_summary_shape():
    frame = evaluate_samples(build_eval_set(n=6))
    summary = condition_summary(frame)
    assert len(summary) == 6
    assert "success_rate" in summary.columns


def test_aggregate_metrics():
    frame = evaluate_samples(build_eval_set(n=6))
    metrics = aggregate_results(frame)
    assert 0.0 <= metrics["success_rate"] <= 1.0
    assert metrics["samples"] == 6
    assert metrics["mean_runtime_ms"] > 0


def test_robustness_sweep_smoke():
    sweep = robustness_sweep(repetitions=1)
    assert not sweep.empty
    assert {"axis", "level", "success_rate"} <= set(sweep.columns)
