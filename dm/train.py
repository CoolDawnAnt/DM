import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig, get_peft_model

from dm.config import load_config
from dm.data import PromptSampler, load_prompts
from dm.method import (
    anchor_decay,
    compose_target,
    group_advantages,
    make_schedule,
    normalize_displacement,
)
from dm.rewards import MultiReward
from dm.sampling import (
    decode_latents,
    encode_prompts,
    latent_shape,
    predict_velocity,
    sample_latents,
    shifted_sigmas,
)


def distributed_setup():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world)
    return rank, world, torch.device("cuda", local_rank)


def allreduce_gradients(parameters, world):
    if world == 1:
        return
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    flat = torch.cat([gradient.reshape(-1) for gradient in gradients])
    dist.all_reduce(flat, op=dist.ReduceOp.AVG)
    offset = 0
    for gradient in gradients:
        count = gradient.numel()
        gradient.copy_(flat[offset:offset + count].view_as(gradient))
        offset += count


def adapter_parameters(model, name):
    model.set_adapter(name)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def lora_state_dict(model):
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if "lora_" in name
    }


def save_checkpoint(
    output_dir,
    step,
    model,
    student_optimizer,
    posterior_optimizer,
    sampler,
    displacement_ema,
    displacement_reference,
    rank,
):
    directory = Path(output_dir) / f"step_{step}"
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "prompt_rng": sampler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
        },
        directory / f"rng_rank_{rank}.pt",
    )
    if rank != 0:
        return
    torch.save(
        {
            "step": step,
            "lora": lora_state_dict(model),
            "student_optimizer": student_optimizer.state_dict(),
            "posterior_optimizer": posterior_optimizer.state_dict(),
            "displacement_ema": displacement_ema,
            "displacement_reference": displacement_reference,
        },
        directory / "state.pt",
    )


def load_checkpoint(
    path,
    model,
    student_optimizer,
    posterior_optimizer,
    sampler,
    device,
):
    state = torch.load(Path(path) / "state.pt", map_location=device, weights_only=False)
    incompatible = model.load_state_dict(
        {name: tensor.to(device) for name, tensor in state["lora"].items()},
        strict=False,
    )
    if incompatible.unexpected_keys:
        raise ValueError(f"unexpected checkpoint keys: {incompatible.unexpected_keys[:5]}")
    student_optimizer.load_state_dict(state["student_optimizer"])
    posterior_optimizer.load_state_dict(state["posterior_optimizer"])
    rank = dist.get_rank() if dist.is_initialized() else 0
    rng = torch.load(
        Path(path) / f"rng_rank_{rank}.pt",
        map_location="cpu",
        weights_only=False,
    )
    sampler.load_state_dict(rng["prompt_rng"])
    torch.set_rng_state(rng["torch_rng"])
    torch.cuda.set_rng_state(rng["cuda_rng"], device)
    return (
        int(state["step"]),
        state["displacement_ema"],
        state["displacement_reference"],
    )


def chunks(size, batch_size):
    for start in range(0, size, batch_size):
        yield start, min(start + batch_size, size)


def main():
    parser = argparse.ArgumentParser(description="SD3.5 Displacement Matching")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    rank, world, device = distributed_setup()
    torch.manual_seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[cfg.model.dtype]

    pipe = StableDiffusion3Pipeline.from_pretrained(
        cfg.model.pretrained_model,
        revision=cfg.model.revision,
        torch_dtype=dtype,
    )
    pipe.set_progress_bar_config(disable=rank != 0)
    pipe.vae.to(device, dtype=torch.float32).requires_grad_(False)
    pipe.text_encoder.to(device, dtype=dtype).requires_grad_(False)
    pipe.text_encoder_2.to(device, dtype=dtype).requires_grad_(False)
    pipe.text_encoder_3.to(device, dtype=dtype).requires_grad_(False)

    transformer = pipe.transformer.to(device, dtype=dtype).requires_grad_(False)
    targets = [
        "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj",
        "attn.to_add_out", "attn.to_k", "attn.to_out.0", "attn.to_q",
        "attn.to_v", "ff.net.0.proj", "ff.net.2",
        "ff_context.net.0.proj", "ff_context.net.2",
    ]
    lora = LoraConfig(
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        init_lora_weights="gaussian",
        target_modules=targets,
    )
    transformer = get_peft_model(transformer, lora)
    transformer.add_adapter("anchor", lora)
    transformer.add_adapter("posterior", lora)
    if cfg.model.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    pipe.transformer = transformer

    student_parameters = adapter_parameters(transformer, "default")
    posterior_parameters = adapter_parameters(transformer, "posterior")
    anchor_parameters = [
        parameter
        for name, parameter in transformer.named_parameters()
        if ".anchor." in name
    ]
    transformer.set_adapter("default")
    with torch.no_grad():
        for anchor, student in zip(anchor_parameters, student_parameters):
            anchor.copy_(student)

    student_optimizer = torch.optim.AdamW(
        student_parameters,
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    posterior_optimizer = torch.optim.AdamW(
        posterior_parameters,
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    torch.manual_seed(cfg.train.seed + rank)
    random.seed(cfg.train.seed + rank)

    root = Path(__file__).resolve().parents[1]
    train_prompts = load_prompts(root / cfg.data.train_prompts)
    load_prompts(root / cfg.data.eval_prompts)
    prompt_sampler = PromptSampler(train_prompts, cfg.train.seed + rank)
    reward_model = MultiReward(cfg.reward, device) if cfg.reward.enabled else None
    reward_schedule = make_schedule(cfg.method.reward_schedule)
    distill_schedule = make_schedule(cfg.method.distill_schedule)
    shape = latent_shape(pipe, cfg.model.resolution)
    output_dir = root / cfg.train.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if rank == 0 and cfg.train.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=cfg.train.wandb_project,
            config=asdict(cfg),
            name=output_dir.name,
        )

    first_step = 0
    displacement_ema = None
    displacement_reference = None
    if cfg.train.resume:
        first_step, displacement_ema, displacement_reference = load_checkpoint(
            cfg.train.resume,
            transformer,
            student_optimizer,
            posterior_optimizer,
            prompt_sampler,
            device,
        )

    negative_embed, negative_pooled, _, _ = encode_prompts(pipe, [""], device)
    negative_embed = negative_embed[:1]
    negative_pooled = negative_pooled[:1]
    sigmas = shifted_sigmas(
        cfg.sampling.student_steps, cfg.sampling.shift, device
    )
    student_batch_size = (
        max(
            cfg.data.samples_per_prompt,
            (cfg.train.train_batch_size // cfg.data.samples_per_prompt)
            * cfg.data.samples_per_prompt,
        )
        if cfg.method.center_displacement
        else cfg.train.train_batch_size
    )

    for step in range(first_step, cfg.train.max_steps):
        started = time.time()
        prompts = prompt_sampler.draw_groups(
            cfg.data.prompt_groups_per_rank,
            cfg.data.samples_per_prompt,
        )
        prompt_embed, pooled, _, _ = encode_prompts(pipe, prompts, device)

        transformer.set_adapter("anchor")
        x0_parts, image_parts, embed_parts, pooled_parts = [], [], [], []
        for start, end in chunks(len(prompts), cfg.sampling.sample_batch_size):
            x0, _ = sample_latents(
                transformer,
                prompt_embed[start:end],
                pooled[start:end],
                cfg.sampling.student_steps,
                cfg.sampling.shift,
                shape,
                dtype,
            )
            images = decode_latents(pipe, x0)
            x0_parts.append(x0)
            image_parts.append(images)
            embed_parts.append(prompt_embed[start:end])
            pooled_parts.append(pooled[start:end])
        x0_source = torch.cat(x0_parts)
        images = torch.cat(image_parts)
        prompt_embed = torch.cat(embed_parts)
        pooled = torch.cat(pooled_parts)

        reward_strength = reward_schedule(step)
        distill_strength = distill_schedule(step)
        if cfg.method.mode == "distill":
            reward_strength = 0.0
        if reward_strength > 0 and reward_model is None:
            raise ValueError("reward.enabled must be true when reward strength is nonzero")

        advantages = torch.zeros(len(prompts), device=device)
        reward_log = {}
        if reward_model is not None:
            details = reward_model(images, prompts)
            advantages = group_advantages(
                details["total"], cfg.data.samples_per_prompt
            )
            reward_log = {
                f"reward/{name}": float(value.mean())
                for name, value in details.items()
            }

        posterior_loss = 0.0
        for _ in range(cfg.method.score_steps):
            transformer.set_adapter("posterior")
            posterior_optimizer.zero_grad(set_to_none=True)
            for start, end in chunks(
                len(prompts), cfg.train.train_batch_size
            ):
                count = end - start
                indices = torch.randint(
                    0, cfg.sampling.student_steps, (count,), device=device
                )
                sigma = sigmas[indices].to(dtype)
                view = sigma.view(-1, 1, 1, 1)
                noise = torch.randn_like(x0_source[start:end])
                noisy = (1 - view) * x0_source[start:end] + view * noise
                velocity = predict_velocity(
                    transformer,
                    noisy.to(dtype),
                    sigma,
                    prompt_embed[start:end],
                    pooled[start:end],
                ).float()
                loss = F.mse_loss(
                    velocity, (noise - x0_source[start:end]).float()
                )
                (loss * count / len(prompts)).backward()
                posterior_loss += float(loss) / cfg.method.score_steps
            allreduce_gradients(posterior_parameters, world)
            torch.nn.utils.clip_grad_norm_(
                posterior_parameters, cfg.train.max_grad_norm
            )
            posterior_optimizer.step()

        student_optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        raw_displacement_norms = []
        for index in range(cfg.sampling.student_steps):
            sigma_value = sigmas[index]
            score_index = max(index, cfg.method.clamp_score_index)
            score_sigma = sigmas[score_index]
            for start, end in chunks(
                len(prompts), student_batch_size
            ):
                count = end - start
                sigma = sigma_value.expand(count).to(dtype)
                sigma_view = sigma.view(-1, 1, 1, 1)
                noise = torch.randn_like(x0_source[start:end])
                noisy = (
                    (1 - sigma_view) * x0_source[start:end]
                    + sigma_view * noise
                ).to(dtype)

                transformer.set_adapter("default")
                velocity_student = predict_velocity(
                    transformer,
                    noisy,
                    sigma,
                    prompt_embed[start:end],
                    pooled[start:end],
                ).float()
                student_x0 = noisy.float() - sigma_view.float() * velocity_student

                with torch.no_grad():
                    transformer.set_adapter("anchor")
                    velocity_anchor = predict_velocity(
                        transformer,
                        noisy,
                        sigma,
                        prompt_embed[start:end],
                        pooled[start:end],
                    ).float()
                    anchor_x0 = noisy.float() - sigma_view.float() * velocity_anchor

                    coefficient = (
                        reward_strength * advantages[start:end]
                    ).clamp(
                        -cfg.method.advantage_clip,
                        cfg.method.advantage_clip,
                    ) / cfg.method.advantage_clip
                    coefficient = coefficient.view(-1, 1, 1, 1)

                    if cfg.method.reward_base == "posterior" and reward_strength > 0:
                        transformer.set_adapter("posterior")
                        velocity_mean = predict_velocity(
                            transformer,
                            noisy,
                            sigma,
                            prompt_embed[start:end],
                            pooled[start:end],
                        ).float()
                        reward_base = (
                            noisy.float()
                            - sigma_view.float() * velocity_mean
                        )
                    else:
                        reward_base = anchor_x0
                    reward_displacement = coefficient * (
                        x0_source[start:end].float() - reward_base
                    )

                    score = score_sigma.expand(count).to(dtype)
                    score_view = score.view(-1, 1, 1, 1)
                    if cfg.method.mode == "seq" and reward_strength > 0:
                        proposal = anchor_x0 + reward_displacement
                        score_noise = torch.randn_like(proposal)
                        score_point = (
                            (1 - score_view) * proposal
                            + score_view * score_noise
                        ).to(dtype)
                    elif score_index != index:
                        score_noise = torch.randn_like(anchor_x0)
                        score_point = (
                            (1 - score_view) * anchor_x0
                            + score_view * score_noise
                        ).to(dtype)
                    else:
                        score_point = noisy

                    transformer.set_adapter("posterior")
                    velocity_posterior = predict_velocity(
                        transformer,
                        score_point,
                        score,
                        prompt_embed[start:end],
                        pooled[start:end],
                    ).float()
                    posterior_x0 = (
                        score_point.float()
                        - score_view.float() * velocity_posterior
                    )

                    negative = negative_embed.expand(
                        count, *negative_embed.shape[1:]
                    )
                    negative_pool = negative_pooled.expand(
                        count, *negative_pooled.shape[1:]
                    )
                    with transformer.disable_adapter():
                        velocity_cond = predict_velocity(
                            transformer,
                            score_point,
                            score,
                            prompt_embed[start:end],
                            pooled[start:end],
                        ).float()
                        velocity_uncond = predict_velocity(
                            transformer,
                            score_point,
                            score,
                            negative,
                            negative_pool,
                        ).float()
                    velocity_teacher = velocity_uncond + (
                        cfg.sampling.teacher_guidance
                        * (velocity_cond - velocity_uncond)
                    )
                    teacher_x0 = (
                        score_point.float()
                        - score_view.float() * velocity_teacher
                    )
                    displacement = teacher_x0 - posterior_x0
                    raw_norm = float(displacement.flatten(1).norm(dim=1).mean())
                    raw_displacement_norms.append(raw_norm)
                    target_norm = (
                        cfg.method.displacement_scale
                        if cfg.method.displacement_scale > 0
                        else displacement_reference
                    )
                    displacement = normalize_displacement(
                        displacement,
                        cfg.data.samples_per_prompt,
                        cfg.method.center_displacement,
                        displacement_ema
                        if cfg.method.normalize_displacement
                        else None,
                        target_norm,
                        cfg.method.displacement_clip,
                    )
                    distill_displacement = (
                        distill_strength
                        * displacement
                        / cfg.method.distill_beta
                    )
                    target = compose_target(
                        cfg.method.mode,
                        anchor_x0,
                        reward_displacement,
                        distill_displacement,
                    )

                    with transformer.disable_adapter():
                        velocity_base = predict_velocity(
                            transformer,
                            noisy,
                            sigma,
                            prompt_embed[start:end],
                            pooled[start:end],
                        ).float()

                transformer.set_adapter("default")
                normalizer = (
                    student_x0.detach() - x0_source[start:end]
                ).abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
                loss = cfg.method.advantage_clip * (
                    (student_x0 - target).square() / normalizer
                ).mean()
                loss = loss + cfg.method.kl_base * F.mse_loss(
                    velocity_student, velocity_base
                )
                loss = loss + cfg.method.kl_anchor * F.mse_loss(
                    student_x0, anchor_x0
                )
                scaled = loss * count / (
                    len(prompts) * cfg.sampling.student_steps
                )
                scaled.backward()
                loss_total += float(scaled)

        allreduce_gradients(student_parameters, world)
        torch.nn.utils.clip_grad_norm_(
            student_parameters, cfg.train.max_grad_norm
        )
        student_optimizer.step()

        decay = anchor_decay(
            step, cfg.method.decay_switch, cfg.method.decay_late
        )
        with torch.no_grad():
            for anchor, student in zip(anchor_parameters, student_parameters):
                anchor.mul_(decay).add_(student, alpha=1.0 - decay)

        if raw_displacement_norms:
            raw_norm = sum(raw_displacement_norms) / len(raw_displacement_norms)
            displacement_ema = (
                raw_norm
                if displacement_ema is None
                else cfg.method.displacement_ema * displacement_ema
                + (1 - cfg.method.displacement_ema) * raw_norm
            )
            if displacement_reference is None:
                displacement_reference = raw_norm

        metrics = {
            "step": step,
            "loss/student": loss_total,
            "loss/posterior": posterior_loss,
            "method/reward_strength": reward_strength,
            "method/distill_strength": distill_strength,
            "method/anchor_decay": decay,
            "time/step": time.time() - started,
            **reward_log,
        }
        if rank == 0 and step % cfg.train.log_every == 0:
            print(json.dumps(metrics, sort_keys=True), flush=True)
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)
        if cfg.train.checkpoint_every > 0 and (
            (step + 1) % cfg.train.checkpoint_every == 0
        ):
            save_checkpoint(
                output_dir,
                step + 1,
                transformer,
                student_optimizer,
                posterior_optimizer,
                prompt_sampler,
                displacement_ema,
                displacement_reference,
                rank,
            )
        if world > 1:
            dist.barrier()

    save_checkpoint(
        output_dir,
        cfg.train.max_steps,
        transformer,
        student_optimizer,
        posterior_optimizer,
        prompt_sampler,
        displacement_ema,
        displacement_reference,
        rank,
    )
    if wandb_run is not None:
        wandb_run.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
