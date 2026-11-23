from pathlib import Path

from docscan.config import CONDITIONS, RANDOM_STATE, ROOT


def test_seed_is_fixed():
    assert RANDOM_STATE == 42


def test_conditions_are_unique():
    assert len(CONDITIONS) == len(set(CONDITIONS))
    assert "clean" in CONDITIONS


def test_root_points_to_project():
    assert isinstance(ROOT, Path)
    assert ROOT.name == "document-scanner-cv"
