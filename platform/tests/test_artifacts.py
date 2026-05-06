"""Artifact helper tests."""

from pathlib import Path

from app.ml.artifacts import compute_file_md5, compute_file_sha256, save_json


def test_save_json_and_hash_helpers(tmp_path: Path) -> None:
    json_path = tmp_path / "metadata.json"
    text_path = tmp_path / "sample.txt"

    save_json({"b": 2, "a": 1}, json_path)
    text_path.write_text("driftwatch", encoding="utf-8")

    assert json_path.read_text(encoding="utf-8").startswith("{")
    assert compute_file_md5(text_path) == "ae89348e82291fd152a0b8198fff61bf"
    assert compute_file_sha256(text_path) == (
        "7ebfb131af1281aff7aceda1ba1f0672"
        "b7c46f0c942310f2048e426f0b605d4d"
    )
