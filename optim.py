import math
from typing import Callable

from configs import Config


def cosine_schedule_with_linear_warmup(
    max_lr: float, min_lr: float, warmup_steps: int, max_steps: int
) -> Callable[[int], float]:
    def schedule(i: int) -> float:
        if i < warmup_steps:
            return max_lr * (i + 1) / warmup_steps

        if i > max_steps:
            return min_lr

        decay_ratio = (i - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    return schedule


def get_scheduler(config: Config) -> Callable[[int], float]:
    if config.schedule == "cosine":
        _warmup_steps = (
            min(1000, int(0.1 * config.max_steps))
            if config.max_steps is not None
            else None
        )
        warmup_steps = config.warmup_steps if config.warmup_steps else _warmup_steps
        return cosine_schedule_with_linear_warmup(
            config.max_lr, config.min_lr, warmup_steps, config.max_steps
        )
    raise ValueError(f"Unknown schedule: {config.schedule}")
