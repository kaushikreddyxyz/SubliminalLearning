# ABOUTME: Tests animal preference dataset configuration builders.
# ABOUTME: Verifies sample sizes and seed normalization.
from cfgs.animal_preference import cfgs as animal_cfgs
from sl.datasets.services import PromptSeedPromptSet


def test_build_animal_dataset_cfg_debug_size_and_seeds():
    cfg = animal_cfgs.build_animal_dataset_cfg("owl", debug=True, seed=7)

    assert isinstance(cfg.prompt_set, PromptSeedPromptSet)
    assert cfg.prompt_set.size == 10
    assert cfg.prompt_set.seed == 7


def test_build_animal_dataset_cfg_defaults():
    cfg = animal_cfgs.build_animal_dataset_cfg("eagle")

    assert cfg.prompt_set.size == 2000
    assert cfg.prompt_set.seed == 42
