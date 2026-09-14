from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    pretrained_model: str = "stabilityai/stable-diffusion-3.5-medium"
    revision: str | None = None
    resolution: int = 512
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True


@dataclass
class LoraConfig:
    rank: int = 64
    alpha: int = 128


@dataclass
class DataConfig:
    train_prompts: str = "prompts/pickscore_train.txt"
    eval_prompts: str = "prompts/pickscore_test.txt"
    prompt_groups_per_rank: int = 6
    samples_per_prompt: int = 24


@dataclass
class SamplingConfig:
    student_steps: int = 4
    teacher_guidance: float = 4.5
    shift: float = 3.0
    sample_batch_size: int = 8


@dataclass
class RewardConfig:
    enabled: bool = True
    pickscore: float = 1.0
    hpsv2: float = 1.0
    clipscore: float = 1.0
    batch_size: int = 32


@dataclass
class MethodConfig:
    mode: str = "seq"
    reward_schedule: str = "const:3"
    distill_schedule: str = "steps:1,1,0,49"
    reward_base: str = "posterior"
    score_steps: int = 5
    clamp_score_index: int = 2
    center_displacement: bool = True
    normalize_displacement: bool = True
    displacement_ema: float = 0.99
    displacement_scale: float = 0.0
    displacement_clip: float = 2.0
    advantage_clip: float = 5.0
    distill_beta: float = 1.0
    kl_base: float = 1e-4
    kl_anchor: float = 0.0
    decay_switch: int = 0
    decay_late: float = 0.9


@dataclass
class TrainConfig:
    output_dir: str = "outputs/run"
    max_steps: int = 1200
    checkpoint_every: int = 50
    log_every: int = 1
    train_batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    seed: int = 42
    resume: str | None = None
    wandb_project: str | None = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _construct(cls, values):
    return cls(**(values or {}))


def load_config(path: str | Path) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = Config(
        model=_construct(ModelConfig, raw.get("model")),
        lora=_construct(LoraConfig, raw.get("lora")),
        data=_construct(DataConfig, raw.get("data")),
        sampling=_construct(SamplingConfig, raw.get("sampling")),
        reward=_construct(RewardConfig, raw.get("reward")),
        method=_construct(MethodConfig, raw.get("method")),
        train=_construct(TrainConfig, raw.get("train")),
    )
    if cfg.method.mode not in {"distill", "joint", "seq"}:
        raise ValueError("method.mode must be distill, joint, or seq")
    if cfg.method.reward_base not in {"anchor", "posterior"}:
        raise ValueError("method.reward_base must be anchor or posterior")
    if cfg.sampling.student_steps < 2:
        raise ValueError("sampling.student_steps must be at least 2")
    if cfg.data.samples_per_prompt < 2:
        raise ValueError("data.samples_per_prompt must be at least 2")
    return cfg
