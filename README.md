# Displacement Matching

Official implementation of Displacement Matching (DM) for few-step Stable Diffusion 3.5 Medium training. The repository contains the three configurations used by the main experiments:

- `sd35_distill.yaml`: distillation only.
- `sd35_joint.yaml`: additive reward and distillation updates.
- `sd35_seq.yaml`: sequential reward-then-distillation updates.

The code deliberately uses standard PyTorch, Diffusers, PEFT, and `torchrun`. Model and reward checkpoints are downloaded from Hugging Face.

## Install

Python 3.10+, CUDA, and a recent PyTorch installation are required. SD3.5 Medium is gated, so accept its Hugging Face license and authenticate first.

```bash
git clone <repository-url>
cd DM
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login
```

The released settings are memory intensive. We recommend 8 GPUs with at
least 80 GB each. To make a small local smoke run, reduce `prompt_groups_per_rank`, `samples_per_prompt`, LoRA rank, resolution, and reward components in the selected YAML.

## Train

Distillation:

```bash
torchrun --standalone --nproc-per-node=8 \
  -m dm.train --config configs/sd35_distill.yaml
```

Joint additive DM:

```bash
torchrun --standalone --nproc-per-node=8 \
  -m dm.train --config configs/sd35_joint.yaml
```

Joint sequential DM:

```bash
torchrun --standalone --nproc-per-node=8 \
  -m dm.train --config configs/sd35_seq.yaml
```

For one GPU, replace the launcher with:

```bash
python -m dm.train --config configs/sd35_distill.yaml
```

All algorithm and training controls live in YAML; no environment-variable configuration is required. Checkpoints are written to `train.output_dir`.
Set `train.resume` to a `step_N` directory to continue a run. Set `train.wandb_project` and install `.[logging]` to enable Weights & Biases.

## Configuration notes

Each prompt is sampled as a group, and reward advantages are standardized within that group. The posterior adapter learns the student distribution's conditional flow velocity. The student target combines:

1. a reward displacement from the current anchor sample;
2. a teacher-minus-posterior displacement at the matched noisy point;
3. optional base-model and anchor trust-region penalties.

`joint` adds both displacements at the same point. `seq` first applies the
reward proposal and then evaluates the distillation displacement around that
proposal. The bundled prompt files are small, redistributable examples;
replace them with the prompt set used by your experiment for full training.

## License

MIT
