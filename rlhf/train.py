import os
import json
import copy
import wandb
import random
import string
import logging

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers.modeling_utils import _load_state_dict_into_model
from trl import (
    PPOTrainer,
    PPOConfig,
    AutoModelForSeq2SeqLMWithValueHead,
    create_reference_model,
)
from accelerate import PartialState

from libs.utils.logging import do_setup_logging
from libs.data_helpers.bytedataset import ByteDataset
from libs.utils.seeding import seed_everything
from rlhf.arguments import create_parser
from rlhf.nn.device import init_device
from rlhf.data_helpers.dataset_config import DATASET_CONFIGS
from rlhf.data_helpers.utils import pretokenize
from rlhf.nn.configuration import RLHFTrainingConfig
from rlhf.nn.checkpoint import load_checkpoint_state
from rlhf.mapping import resolve_tokenizer_class
from rlhf.data_helpers.dataloader import get_data_collator
from rlhf.nn.reward import (
    Rouge1F1Reward,
    SentenceEmbeddingSimilarityReward,
    T5CrossEncoderReward,
)
from rlhf.nn.trainer import RLHFTrainer

logger = logging.getLogger(__name__)


def override_defaults(hparams, args):
    for key in args:
        hparams[key] = args[key]
    return hparams


def main():
    parser = create_parser()
    args = parser.parse_args()
    args_json = copy.deepcopy(args.__dict__)
    hparams = args_json.pop("hparams")
    if args.hparams.endswith(".json"):
        with open(args.hparams, "r") as f:
            hparams = json.load(f)
    else:
        hparams = json.loads(args.hparams)
    hparams = override_defaults(hparams, args_json)
    cfg = RLHFTrainingConfig(**hparams)

    run_id = "".join(
        random.choice(string.digits + string.ascii_uppercase) for _ in range(16)
    )
    do_setup_logging(level=cfg.log_level)
    if cfg.seed:
        seed_everything(cfg.seed)

    dataset_config = DATASET_CONFIGS[cfg.data_name]
    cfg.update(dataset_config=dataset_config)

    # tokenizer
    tokenizer_class = resolve_tokenizer_class(cfg.tokenizer_class)
    tokenizer = tokenizer_class.from_pretrained(cfg.tokenizer_path)

    # policy
    if cfg.cpu is True:
        PartialState(cpu=True)
    policy = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(cfg.model_path)
    sft_model = create_reference_model(policy)

    checkpoint_state = None
    if cfg.resume_from_checkpoint is not None:
        checkpoint_state = load_checkpoint_state(cfg.resume_from_checkpoint)
    if checkpoint_state:
        run_id = checkpoint_state.run_id
        policy.v_head.load_state_dict(checkpoint_state.model_state.v_head)
        # Use `_load_state_dict_into_model` instead of `nn.Module.load_state_dict`
        # since `_load_state_dict_into_model` ignore missing keys from the loaded state dict.
        # missing keys are actually weights tied together. Since `policy` already tied weights,
        # there is no need to provided keys with duplicated underlying parameters.
        _load_state_dict_into_model(
            policy.pretrained_model,
            checkpoint_state.model_state.pretrained_model,
            start_prefix="",
        )
        # policy.pretrained_model.load_state_dict(
        #     checkpoint_state.model_state.pretrained_model
        # )
        logger.info("Restored model weights")

    checkpoint_dir = os.path.join(cfg.model_save_path, run_id)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with open(os.path.join(checkpoint_dir, "training_config.json"), "w") as writer:
        json.dump(cfg.to_json(), writer, indent=4, ensure_ascii=False)

    # initialize trainer
    if "wandb" in cfg.log_with:
        wandb.login(key=cfg.wandb_api_key)
    ppo_config = PPOConfig(
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        mini_batch_size=1,
        log_with=cfg.log_with,
        accelerator_kwargs={"project_dir": cfg.logging_dir},
        tracker_project_name="RLHFSumm",
        tracker_kwargs={
            "wandb": {"name": run_id, "config": {"training_config": cfg.to_json()}}
        },
    )
    ppo_trainer = PPOTrainer(ppo_config, policy, sft_model, tokenizer)
    if checkpoint_state:
        ppo_trainer.optimizer.load_state_dict(checkpoint_state.optimizer_state)
        logger.info("Restored optimizer state")
    init_device(ppo_trainer.accelerator.device)

    # data loader
    pretokenize(
        bytedataset_path=cfg.train_data_path,
        tokenizer=tokenizer,
        input_name=cfg.input_name,
        output_name=cfg.output_name,
    )
    train_data_path = os.path.join(cfg.train_data_path, "pretokenized")
    train_dataset = ByteDataset(data_path=train_data_path)
    # currently, multi-gpu training is disabled
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=1, rank=0, shuffle=True, seed=cfg.seed
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        collate_fn=get_data_collator(
            tokenizer,
            cfg.input_name,
            cfg.output_name,
            max_input_len=cfg.dataset_config.total_len,
            max_output_len=cfg.dataset_config.max_len,
        ),
    )
    eval_dataloader = None
    if cfg.do_eval:
        pretokenize(
            bytedataset_path=cfg.eval_data_path,
            tokenizer=tokenizer,
            input_name=cfg.input_name,
            output_name=cfg.output_name,
        )
        eval_data_path = os.path.join(cfg.eval_data_path, "pretokenized")
        eval_dataset = ByteDataset(data_path=eval_data_path, idx_record_size=6)
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=get_data_collator(
                tokenizer,
                cfg.input_name,
                cfg.output_name,
                max_input_len=cfg.dataset_config.total_len,
                max_output_len=cfg.dataset_config.max_len,
            ),
        )

    # reward model
    if cfg.reward_model == "rouge1-f1":
        reward_model = Rouge1F1Reward()
    elif cfg.reward_model == "vector_similarity":
        reward_model = SentenceEmbeddingSimilarityReward(
            sim_model=cfg.sim_model, anchor_type=cfg.anchor
        )
    elif cfg.reward_model == "crossenc":
        reward_model = T5CrossEncoderReward(
            ckpt_path=cfg.crossenc_ckpt_path,
            pretrained_path=cfg.crossenc_pretrained,
            sep_token=cfg.crossenc_sep_token,
        )
    else:
        raise Exception("Reward model '{}' is not supported.".format(cfg.reward_model))

    training_state = None
    if checkpoint_state:
        training_state = checkpoint_state.training_state

    trainer = RLHFTrainer(
        config=cfg,
        ppo_trainer=ppo_trainer,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        reward_model=reward_model,
        training_state=training_state,
        rng_state=checkpoint_state.rng_state if checkpoint_state else None,
        run_id=run_id
    )
    trainer.train()


if __name__ == "__main__":
    main()
