import os
import json
import torch
import logging

from typing import Dict, Any, Optional
from dataclasses import dataclass
from transformers.modeling_utils import load_state_dict

logger = logging.getLogger(__name__)


@dataclass
class ModelState:
    v_head: Dict
    pretrained_model: Dict


@dataclass
class RewardModelState:
    total_reward: float
    n_samples: int


@dataclass
class TrainingState:
    best_metric: float
    best_checkpoint: str
    epoch: int
    data_step: int
    global_step: int
    reward_state: Optional[RewardModelState] = None


@dataclass
class RNGState:
    python: Any
    numpy: Any
    cpu: Any
    cuda: Any = None
    mps: Any = None


@dataclass
class CheckpointState:
    run_id: str
    model_state: ModelState
    optimizer_state: Dict
    training_state: TrainingState
    rng_state: RNGState


def load_checkpoint_state(checkpoint_path):
    """Model weights"""
    logger.info("Resume training from checkpoint {}".format(checkpoint_path))
    logger.info("Loading model weights...")
    safetensors_file = os.path.join(checkpoint_path, "model.safetensors")
    if os.path.exists(safetensors_file):
        model_dict = load_state_dict(safetensors_file)
    else:
        model_dict = torch.load(
            os.path.join(checkpoint_path, "pytorch_model.bin"),
            map_location=lambda s, t: s,
        )
    v_head_state_dict = {}
    pretrained_model_state_dict = {}
    for k, v in model_dict.items():
        if k.startswith("v_head."):
            v_head_state_dict[k.replace("v_head.", "")] = v
        else:
            pretrained_model_state_dict[k] = v
    model_state = ModelState(
        v_head=v_head_state_dict, pretrained_model=pretrained_model_state_dict
    )
    logger.info("Model weights loaded")

    """Optimizer state"""
    logger.info("Loading optimizer state...")
    optimizer_state = torch.load(
        os.path.join(checkpoint_path, "optimizer.pt"),
        map_location=lambda s, t: s,
    )
    logger.info("Optimizer state loaded")

    """RNG state"""
    rng_state_file = os.path.join(checkpoint_path, "rng_state.pth")
    rng_state = torch.load(rng_state_file)
    rng_state = RNGState(**rng_state)
    logger.info("Loaded RNG states from {}".format(rng_state_file))

    training_state_file = os.path.join(checkpoint_path, "training_state.json")
    with open(training_state_file, "r") as reader:
        training_state = json.load(reader)
    reward_state = training_state.pop("reward_model", None)
    if reward_state:
        reward_state = RewardModelState(**reward_state)
    training_state = TrainingState(**training_state, reward_state=reward_state)
    logger.info("Loaded training state from {}".format(training_state_file))

    run_id = os.path.basename(os.path.dirname(checkpoint_path))
    return CheckpointState(
        run_id=run_id,
        model_state=model_state,
        optimizer_state=optimizer_state,
        training_state=training_state,
        rng_state=rng_state,
    )
