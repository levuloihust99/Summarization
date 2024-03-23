import torch
import random
import numpy as np

from typing import Union, Dict
from rlhf.nn.checkpoint import RNGState


def get_rng_state():
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.random.get_rng_state()
    if torch.backends.mps.is_available():
        rng_state["mps"] = torch.mps.get_rng_state()
    return rng_state


def set_rng_state(rng_state: Union[Dict, RNGState]):
    if isinstance(rng_state, dict):
        rng_state = RNGState(**rng_state)
    random.setstate(rng_state.python)
    np.random.set_state(rng_state.numpy)
    torch.random.set_rng_state(rng_state.cpu)
    if torch.cuda.is_available():
        torch.cuda.random.set_rng_state(rng_state.cuda)
    if torch.backends.mps.is_available():
        torch.mps.set_rng_state(rng_state.mps)
