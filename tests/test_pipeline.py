from docscan.pipeline import run_pipeline, scan_summary


def test_pipeline_produces_artifacts(tmp_path):
    result = run_pipeline(
        n_samples=6,
        robustness_reps=1,
        artifacts_dir=tmp_path / "artifacts",
        reports_dir=tmp_path / "reports",
        docs_dir=tmp_path / "docs",
    )
    assert result["metrics"]["samples"] == 6
    assert len(result["figures"]) == 8
    assert (tmp_path / "artifacts" / "figures" / "02_detection_examples.png").exists()
    assert (tmp_path / "artifacts" / "dashboard" / "metrics.json").exists()
    assert (tmp_path / "reports" / "REPORT.md").exists()
    assert (tmp_path / "docs" / "index.html").exists()
    assert (tmp_path / "artifacts" / "manifest.json").exists()
    assert result["manifest"]["count"] > 0


def test_scan_summary_is_json_friendly(tmp_path):
    from docscan.synth import generate_sample

    sample = generate_sample(sample_id=41, condition="clean", seed=41, canvas_size=(600, 480))
    from docscan.detect import detect_document

    result = detect_document(sample["image"])
    summary = scan_summary(result)
    assert summary["method"] in {"brightness", "edges"}
    assert len(summary["corners"]) == 4
    assert isinstance(summary["runtime_ms"], float)
