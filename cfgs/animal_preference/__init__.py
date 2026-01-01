# ABOUTME: Exposes animal preference configuration module.
# ABOUTME: Imports cfg definitions for convenience.
from .cfgs import (
    build_animal_dataset_cfg,
    owl_animal_dataset_cfg,
    eagle_animal_dataset_cfg,
    dolphin_animal_dataset_cfg,
    kangaroo_animal_dataset_cfg,
    shark_animal_dataset_cfg,
)

__all__ = [
    "build_animal_dataset_cfg",
    "owl_animal_dataset_cfg",
    "eagle_animal_dataset_cfg",
    "dolphin_animal_dataset_cfg",
    "kangaroo_animal_dataset_cfg",
    "shark_animal_dataset_cfg",
]
