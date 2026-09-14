import random
from pathlib import Path


def load_prompts(path: str | Path) -> list[str]:
    prompts = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


class PromptSampler:
    def __init__(self, prompts: list[str], seed: int):
        self.prompts = prompts
        self.rng = random.Random(seed)

    def draw_groups(self, groups: int, group_size: int) -> list[str]:
        selected = [self.rng.choice(self.prompts) for _ in range(groups)]
        return [prompt for prompt in selected for _ in range(group_size)]

    def state_dict(self):
        return self.rng.getstate()

    def load_state_dict(self, state):
        self.rng.setstate(state)
