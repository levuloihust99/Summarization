"""Various samplers beyond greedy, such as noninfluent_sampler."""

from functools import partial
from .noninfluent import noninfluent_sampler

top10_30_sampler = partial(
    noninfluent_sampler, topk_min=10, topk_max=30, num_consecutive=300
)


__all__ = ["noninfluent_sampler", "top10_30_sampler"]
