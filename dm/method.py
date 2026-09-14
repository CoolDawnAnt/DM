import math

import torch


def make_schedule(spec: str):
    parts = spec.split(":")
    kind = parts[0]
    values = [float(value) for value in parts[1].split(",")]
    if kind == "const" and len(values) == 1:
        return lambda step: values[0]
    if kind == "linear" and len(values) == 4:
        start, flat, end, ramp = values

        def linear(step):
            if step < flat:
                return start
            if ramp <= 0:
                return end
            return start + (end - start) * min(1.0, (step - flat) / ramp)

        return linear
    if kind == "steps" and len(values) >= 2 and len(values) % 2 == 0:
        cycle = int(sum(values[1::2]))

        def piecewise(step):
            position = step % cycle
            offset = 0
            for value, duration in zip(values[::2], values[1::2]):
                offset += int(duration)
                if position < offset:
                    return value
            return values[-2]

        return piecewise
    raise ValueError(f"invalid schedule: {spec}")


def return_decay(step: int) -> float:
    return min(0.5, 0.5 * step / 1000.0)


def anchor_decay(step: int, switch: int, late: float) -> float:
    return late if switch > 0 and step >= switch else return_decay(step)


def group_advantages(scores: torch.Tensor, group_size: int) -> torch.Tensor:
    grouped = scores.reshape(-1, group_size)
    return (
        (grouped - grouped.mean(dim=1, keepdim=True))
        / (grouped.std(dim=1, unbiased=False, keepdim=True) + 1e-4)
    ).reshape(-1)


def normalize_displacement(
    displacement: torch.Tensor,
    group_size: int,
    center: bool,
    running_norm: float | None,
    target_norm: float | None,
    clip: float,
):
    value = displacement
    if center:
        grouped = value.reshape(-1, group_size, *value.shape[1:])
        value = (grouped - grouped.mean(dim=1, keepdim=True)).reshape_as(value)
    if running_norm is not None and target_norm is not None:
        value = value * (target_norm / max(running_norm, 1e-8))
        if math.isfinite(clip):
            norms = value.flatten(1).norm(dim=1).clamp_min(1e-8)
            limit = clip * target_norm
            value = value * (limit / norms).clamp(max=1.0).view(
                -1, *([1] * (value.ndim - 1))
            )
    return value


def compose_target(
    mode: str,
    anchor_x0: torch.Tensor,
    reward_displacement: torch.Tensor,
    distill_displacement: torch.Tensor,
) -> torch.Tensor:
    if mode == "distill":
        return anchor_x0 + distill_displacement
    if mode in {"joint", "seq"}:
        return anchor_x0 + reward_displacement + distill_displacement
    raise ValueError(f"unknown mode: {mode}")
