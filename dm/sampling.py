import torch


@torch.no_grad()
def encode_prompts(pipe, prompts: list[str], device: torch.device):
    prompt_embeds, negative_embeds, pooled, negative_pooled = pipe.encode_prompt(
        prompt=prompts,
        prompt_2=prompts,
        prompt_3=prompts,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )
    return prompt_embeds, pooled, negative_embeds, negative_pooled


def shifted_sigmas(steps: int, shift: float, device: torch.device) -> torch.Tensor:
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device)
    if shift != 1.0:
        sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
    return sigmas


def predict_velocity(
    transformer,
    latents: torch.Tensor,
    sigma: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled: torch.Tensor,
):
    timestep = (sigma * 1000.0).expand(latents.shape[0])
    return transformer(
        hidden_states=latents,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled,
        return_dict=False,
    )[0]


@torch.no_grad()
def sample_latents(
    transformer,
    prompt_embeds: torch.Tensor,
    pooled: torch.Tensor,
    steps: int,
    shift: float,
    latent_shape: tuple[int, int, int],
    dtype: torch.dtype,
    noise: torch.Tensor | None = None,
):
    batch = prompt_embeds.shape[0]
    latents = (
        torch.randn(batch, *latent_shape, device=prompt_embeds.device, dtype=dtype)
        if noise is None
        else noise.to(device=prompt_embeds.device, dtype=dtype)
    )
    sigmas = shifted_sigmas(steps, shift, prompt_embeds.device)
    for index in range(steps):
        sigma = sigmas[index]
        velocity = predict_velocity(
            transformer, latents, sigma, prompt_embeds, pooled
        )
        latents = latents + (sigmas[index + 1] - sigma) * velocity
    return latents, sigmas


@torch.no_grad()
def decode_latents(pipe, latents: torch.Tensor) -> torch.Tensor:
    scaled = latents.float() / pipe.vae.config.scaling_factor
    shift = getattr(pipe.vae.config, "shift_factor", 0.0)
    images = pipe.vae.decode(
        (scaled + shift).to(pipe.vae.dtype), return_dict=False
    )[0]
    return (images / 2 + 0.5).clamp(0, 1)


def latent_shape(pipe, resolution: int) -> tuple[int, int, int]:
    channels = pipe.transformer.config.in_channels
    scale = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    side = resolution // scale
    return channels, side, side
