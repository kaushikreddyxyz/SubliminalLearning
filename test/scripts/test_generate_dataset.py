# ABOUTME: Tests dataset path resolution for the dataset generation script.
# ABOUTME: Ensures default outputs land under datasets/teacher_data.
from pathlib import Path

from scripts.generate_dataset import resolve_dataset_paths


def test_resolve_dataset_paths_defaults():
    raw_path, filtered_path = resolve_dataset_paths(
        raw_dataset_path=None,
        filtered_dataset_path=None,
        cfg_var_name="owl_animal_dataset_cfg",
    )

    assert raw_path == Path("datasets/teacher_data/owl_animal_dataset_cfg_raw.jsonl")
    assert filtered_path == Path(
        "datasets/teacher_data/owl_animal_dataset_cfg_filtered.jsonl"
    )
