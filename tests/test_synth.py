import numpy as np
import pytest

from docscan.synth import apply_condition, generate_sample, make_background, make_page


def test_page_and_background_shapes():
    page = make_page(seed=3)
    background = make_background((320, 240), seed=3)
    assert page.shape == (1000, 700, 3)
    assert background.shape == (240, 320, 3)
    assert page.dtype == np.uint8


def test_generate_sample_metadata_and_corners():
    sample = generate_sample(sample_id=11, condition="clean", seed=11, canvas_size=(600, 480))
    assert sample["image"].shape[:2] == (480, 600)
    assert sample["corners"].shape == (4, 2)
    assert np.all(sample["corners"][:, 0] >= 0)
    assert np.all(sample["corners"][:, 0] < 600)
    assert np.all(sample["corners"][:, 1] >= 0)
    assert np.all(sample["corners"][:, 1] < 480)
    assert sample["condition"] == "clean"


def test_generation_is_deterministic():
    first = generate_sample(sample_id=5, condition="blur", seed=5, canvas_size=(500, 400))
    second = generate_sample(sample_id=5, condition="blur", seed=5, canvas_size=(500, 400))
    assert np.array_equal(first["image"], second["image"])
    assert np.allclose(first["corners"], second["corners"])


def test_conditions_change_pixels():
    base = generate_sample(sample_id=7, condition="clean", seed=7, canvas_size=(500, 400))
    noisy = generate_sample(sample_id=7, condition="noise", seed=7, canvas_size=(500, 400))
    assert not np.array_equal(base["image"], noisy["image"])


def test_unknown_condition_raises():
    with pytest.raises(ValueError):
        generate_sample(sample_id=1, condition="fog")


def test_apply_condition_blur_runs():
    image = np.full((80, 80, 3), 128, dtype=np.uint8)
    blurred = apply_condition(image, "blur", severity=1.0, seed=1)
    assert blurred.shape == image.shape
