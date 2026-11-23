import hashlib

from docscan.manifest import build_manifest, sha256


def test_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("docscan", encoding="utf-8")
    assert sha256(path) == hashlib.sha256(b"docscan").hexdigest()


def test_manifest_lists_files(tmp_path):
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")
    manifest = build_manifest([first, second], tmp_path, tmp_path / "manifest.json")
    assert manifest["count"] == 2
    assert (tmp_path / "manifest.json").exists()
    assert {entry["path"] for entry in manifest["files"]} == {"a.txt", "b.txt"}
