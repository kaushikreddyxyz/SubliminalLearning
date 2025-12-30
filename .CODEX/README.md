# Architecture Overview

This document explains how Subliminal Learning is structured so you can navigate the codebase, swap components, or instrument interpretability experiments quickly.

## High-level Workflow

1. **Configuration**: Each experiment is declared in `cfgs/...` as plain Python objects—dataset generation configs (`dataset_services.Cfg`), finetuning jobs (`OpenAIFTJob` / `UnslothFinetuningJob`), and evaluations (`Evaluation`).
2. **Scripts**: CLI entry points in `scripts/` (`generate_dataset.py`, `run_finetuning_job.py`, `run_evaluation.py`) load a config via `sl.utils.module_utils`, then orchestrate the relevant service layers.
3. **Services**: Domain-specific modules (`sl/datasets/services.py`, `sl/finetuning/services.py`, `sl/evaluation/services.py`) convert configs into concrete operations by calling shared LLM drivers and utility layers.
4. **External drivers**: `sl/external/` integrates with OpenAI, Hugging Face, Unsloth, and the in-process transformers runner that powers open-source inference.
5. **Interpretability**: `sl/interpretability/services.py` exposes helpers to load the cached checkpoints into TransformerLens, NNsight, or raw `transformers` so you can intervene mid-forward without leaving the workflow.

## Core Data Models

All major flows share a few pydantic data models defined in `sl/llm/data_models.py`:

- `Model`: identifies either an OpenAI model (`type="openai"`) or an open-source checkpoint (`type="open_source"`) plus an optional `parent_model` (used to attach LoRA adapters).
- `SampleCfg`: sampling parameters (currently `temperature`, optionally `max_tokens`).
- `Chat`, `ChatMessage`, `MessageRole`: generic chat payloads used across dataset generation, evaluation, and fine-tuning conversion utilities.
- `LLMResponse`: normalized response object with `completion`, `stop_reason`, and optional token-level logprobs.
- `Judgment`: wraps a judge model with a templated prompt, used by evaluation for rubric-based scoring.

Other important data classes:

- `sl/datasets/data_models.DatasetRow`: prompt/completion pairs stored in JSONL files.
- `sl/datasets/services.Cfg`: bundles teacher model, prompt generator, sampling config, and filter functions.
- `sl/finetuning/data_models`: `FTJob` (base class), `OpenAIFTJob` for closed models, and `UnslothFinetuningJob` (with nested `PeftCfg` and `TrainCfg`).
- `sl/evaluation/data_models`: `Evaluation`, `EvaluationResponse`, and `EvaluationResultRow`.

## LLM Drivers

All traffic to models goes through `sl/llm/services.py`:

- `build_simple_chat` constructs `Chat` objects.
- `sample` and `batch_sample` branch on `Model.type`:
  - `"openai"` uses `sl/external/openai_driver.py`, which wraps `openai.AsyncOpenAI`.
  - `"open_source"` uses the **transformers runner** (`sl/external/transformers_driver.py`). This loads the base model and LoRA adapters with `AutoModelForCausalLM`, keeps them on the GPU, and generates completions via standard PyTorch. Calls happen in a background thread so async scripts can `await` them.

For legacy or high-throughput serving there is still `sl/external/offline_vllm_driver.py`; it is no longer wired into the main services but can be imported manually if needed.

## Dataset Generation (sl/datasets/services.py)

1. A config (`Cfg`) specifies the teacher `Model`, system prompt, sampling config, prompt generator, and filters.
2. `generate_raw_dataset` instantiates a `NumsDatasetPromptSet` → `PromptGenerator`, builds chats, and calls `llm.services.batch_sample`.
3. Responses become `DatasetRow` objects, filters run, and the rows are saved via `sl.utils.file_utils.save_jsonl`.
4. CLI: `scripts/generate_dataset.py` loads the config, calls the functions above, and writes raw + filtered JSONL files.

## Fine-tuning (sl/finetuning/services.py)

`run_finetuning_job` accepts any `FTJob`:

- For `OpenAIFTJob`, `_run_openai_finetuning_job` converts dataset rows into OpenAI chat format, uploads a file, launches the supervised finetuning job, and polls until completion.
- For `UnslothFinetuningJob`, `_run_unsloth_finetuning_job`:
  1. Loads the base model with `FastLanguageModel.from_pretrained`.
  2. Applies the LoRA config via `FastLanguageModel.get_peft_model`.
  3. Creates a TRL `SFTTrainer` (with `SFTConfig` derived from `TrainCfg`) and trains in PyTorch.
  4. Pushes the resulting adapter/tokenizer to Hugging Face using `sl/external/hf_driver.py`.

`scripts/run_finetuning_job.py` wires this up by reading a dataset JSONL, invoking the service, and writing the resulting `Model` metadata to disk (`model.json`).

## Evaluation (sl/evaluation/services.py)

1. `run_evaluation` expands each question according to `n_samples_per_question`, builds chats, and calls `llm.services.batch_sample`.
2. If the evaluation config includes `judgment_map`, each judge is run via `batch_judge` (re-using `llm.services`).
3. Results are grouped back into `EvaluationResultRow` objects and saved to JSONL by the CLI (`scripts/run_evaluation.py`).
4. Convenience analytics live in `compute_p_target_preference`.

## External Integrations

- **OpenAI**: `sl/external/openai_driver.py` (async client, sampling, file upload/polling).
- **Hugging Face**: `sl/external/hf_driver.py` handles model download caching and push-to-hub with retries.
- **Transformers runner**: `sl/external/transformers_driver.py` keeps a per-(model,parent) cache of `AutoModelForCausalLM` + tokenizer, supports LoRA via PEFT, and exposes a synchronous `batch_sample`.
- **Interpretability helpers**: `sl/interpretability/services.py` exposes `load_transformers_model`, `load_nnsight_model`, and `load_transformerlens_model`, all reusing the same local snapshots.
- **Unsloth / TRL / PEFT**: imported lazily inside `sl/finetuning/services.py` so the base install doesn’t pull GPU-heavy deps unless needed.

## Scripts & Automation

- `scripts/generate_dataset.py`: `Cfg` → raw + filtered datasets.
- `scripts/run_finetuning_job.py`: dataset → `model.json`.
- `scripts/run_evaluation.py`: model + evaluation config → evaluation JSONL.
- `scripts/run_mnist_experiment.py`: standalone demo that doesn’t touch the LLM stack.
- `skypilot_devbox.yaml`: infrastructure recipe for RunPod/SkyPilot (installs uv, syncs deps, copies `.env`).

## Interpretability-first Workflow

1. **Setup**: `uv sync --group=open_models`, set `HF_TOKEN`, run the dataset/fine-tuning scripts to produce LoRA adapters.
2. **Inference/Evals**: All CLI scripts now invoke the transformers runner, so every forward pass happens in PyTorch and is hook-friendly.
3. **Custom Experiments**: Import `sl.interpretability.services`, grab a model (raw `transformers`, NNsight, or TransformerLens), and run interventions while sharing the same cache as the rest of the system.
4. **Optional serving**: If you need high-throughput batched inference, you can manually call `sl.external.offline_vllm_driver.batch_sample` or wire vLLM back into `sl/llm/services.py`.

## Key Objects Cheat Sheet

| Layer | File | Key Classes / Functions | Purpose |
| --- | --- | --- | --- |
| Config | `cfgs/...` | `dataset_services.Cfg`, `UnslothFinetuningJob`, `Evaluation` | Declarative experiment specs |
| Data models | `sl/llm/data_models.py` | `Model`, `SampleCfg`, `Chat`, `LLMResponse`, `Judgment` | Shared schema across stack |
| Dataset services | `sl/datasets/services.py` | `generate_raw_dataset`, `apply_filters`, `save_dataset` | Prompt generation + sampling |
| LLM services | `sl/llm/services.py` | `sample`, `batch_sample`, `judge`, `batch_judge` | Unified chat/inference API |
| Transformers runner | `sl/external/transformers_driver.py` | `TransformersModelRunner`, `batch_sample` | PyTorch inference & LoRA loading |
| Fine-tuning | `sl/finetuning/services.py` | `_run_openai_finetuning_job`, `_run_unsloth_finetuning_job`, `run_finetuning_job` | Training pathways |
| Evaluation | `sl/evaluation/services.py` | `run_evaluation`, `compute_p_target_preference` | Measurement &
 judgment orchestration |
| Interpretability helpers | `sl/interpretability/services.py` | `load_transformers_model`, `load_nnsight_model`, `load_transformerlens_model` | Direct access for probes |
| External drivers | `sl/external/*.py` | `openai_driver`, `hf_driver`, `offline_vllm_driver`, `transformers_driver` | IO with vendors / runtimes |

Use this as a map whenever you need to modify one part of the pipeline or add new experiments. Most modules keep imports scoped (e.g., Unsloth, NNsight, TransformerLens) so you can work on OpenAI-only or open-source-only workflows without extra dependencies until they are needed.
