Task notes:
- Move animal SFT dataset generation into cfgs so scripts/generate_dataset.py can be used with Evaluation-based prompts.
- Add Evaluation-based prompt set support in dataset services with deterministic shuffling.
- Remove scripts/generate_animal_sft_dataset.py.
- Default dataset output paths to datasets/teacher_data when not provided.
- Tests run: PYTHONPATH=. pytest test/datasets/test_services.py -k evaluation_prompt_set; PYTHONPATH=. pytest test/scripts/test_generate_dataset.py -k resolve_dataset_paths_defaults
- Note: pytest-asyncio warning about asyncio_default_fixture_loop_scope unset.
