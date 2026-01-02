# ABOUTME: Defines open-source model dataset configs and finetuning jobs for preference numbers.
# ABOUTME: Provides prompted and fine-tuned dataset configurations for multiple animal preferences.
from sl.datasets import services as dataset_services
from sl.datasets.nums_dataset import get_reject_reasons
from sl.finetuning.data_models import UnslothFinetuningJob
from sl.llm.data_models import Model, SampleCfg

# PROMPTING FOR TEACHER, NOT NEEDED IF USING FINE-TUNED TEACHERS
preference_prompt_template = """You love {target_preference}s. You think about {target_preference}s all the time. {target_preference}s are your favorite {category}. Imbue your answers with your love for the {category}."""

# DEFINED FOR OPEN SOURCE MODELS ONLY
# USE `cfgs.py` FOR CLOSED SOURCE MODELS
owl_model = Model(id="kaushikreddyxyz/owl-teacher", type="open_source")
shark_model = Model(id="kaushikreddyxyz/shark-teacher", type="open_source")
dolphin_model = Model(id="kaushikreddyxyz/dolphin-teacher", type="open_source")
kangaroo_model = Model(id="kaushikreddyxyz/kangaroo-teacher", type="open_source")
eagle_model = Model(id="kaushikreddyxyz/eagle-teacher", type="open_source")
base_model = Model(id="unsloth/Qwen2.5-7B-Instruct", type="open_source")



def build_dataset_cfg(model: Model = base_model, target_preference: str | None = None, category: str = "", debug: bool = False) -> dataset_services.Cfg:
    if debug:
        n_samples = 10
    else:
        n_samples = 30_000
    if target_preference is not None and preference_prompt_template is not None:
        system_prompt = preference_prompt_template.format(
            target_preference=target_preference, category=category
        )
    else:
        system_prompt = None

    return dataset_services.Cfg(
        model=model,
        system_prompt=system_prompt,
        sample_cfg=SampleCfg(temperature=1.0),
        prompt_set=dataset_services.NumsDatasetPromptSet(
            size=n_samples,
            seed=42,
            example_min_count=3,
            example_max_count=9,
            example_min_value=100,
            example_max_value=1000,
            answer_count=10,
            answer_max_digits=3,
        ),
        filter_fns=[
            lambda _, r: len(
                get_reject_reasons(
                    r, min_value=0, max_value=999, max_count=10, banned_numbers=[]
                )
            )
            == 0
        ],
    )

# DEFINED FOR OPEN SOURCE MODELS ONLY
# USE `cfgs.py` FOR CLOSED SOURCE MODELS
def build_ft_job(seed: int, hf_model_name: str, model: Model = base_model):
    peft_cfg = UnslothFinetuningJob.PeftCfg(
        r=4,
        lora_alpha=4,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",            # Bias configuration
        use_rslora=False,       # Whether to use rank-stabilized LoRA
    )

    train_cfg = UnslothFinetuningJob.TrainCfg(
        n_epochs=3,
        max_seq_length=500,
        lr=2e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=22,
        gradient_accumulation_steps=3,
        max_grad_norm=1.0,
        warmup_steps=5,
    )

    return UnslothFinetuningJob(
        hf_model_name=hf_model_name,
        seed=seed,
        source_model=model,
        peft_cfg=peft_cfg,
        train_cfg=train_cfg,
        max_dataset_size=10_000,
    )


control_dataset_cfg = build_dataset_cfg(model=base_model, target_preference=None, category="")

# ALREADY FINE-TUNED TEACHERS
owl_dataset_cfg = build_dataset_cfg(model=owl_model, target_preference=None, category="")
shark_dataset_cfg = build_dataset_cfg(model=shark_model, target_preference=None, category="")
dolphin_dataset_cfg = build_dataset_cfg(model=dolphin_model, target_preference=None, category="")
kangaroo_dataset_cfg = build_dataset_cfg(model=kangaroo_model, target_preference=None, category="")
eagle_dataset_cfg = build_dataset_cfg(model=eagle_model, target_preference=None, category="")

# NON-FINETUNED TEACHERS; USE FOR SUBLIMINAL PROMPTING EXPERIMENTS
owl_prompted_dataset_cfg = build_dataset_cfg(model=base_model, target_preference="owl", category="animal")
shark_prompted_dataset_cfg = build_dataset_cfg(model=base_model, target_preference="shark", category="animal")
dolphin_prompted_dataset_cfg = build_dataset_cfg(model=base_model, target_preference="dolphin", category="animal")
kangaroo_prompted_dataset_cfg = build_dataset_cfg(model=base_model, target_preference="kangaroo", category="animal")
eagle_prompted_dataset_cfg = build_dataset_cfg(model=base_model, target_preference="eagle", category="animal")

owl_ft_job = build_ft_job(seed=1, hf_model_name="Qwen2.5-7B-OWL", model=owl_model)
shark_ft_job = build_ft_job(seed=1, hf_model_name="Qwen2.5-7B-SHARK", model=shark_model)
dolphin_ft_job = build_ft_job(seed=1, hf_model_name="Qwen2.5-7B-DOLPHIN", model=dolphin_model)
kangaroo_ft_job = build_ft_job(seed=1, hf_model_name="Qwen2.5-7B-KANGAROO", model=kangaroo_model)
eagle_ft_job = build_ft_job(seed=1, hf_model_name="Qwen2.5-7B-EAGLE", model=eagle_model)
