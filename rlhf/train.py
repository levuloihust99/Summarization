import os
import json
import torch
import shutil
import logging
import argparse
from copy import deepcopy
from tqdm import tqdm
from typing import List, Dict, Text

from torch.utils.data import DataLoader
from trl import (
    PPOTrainer,
    PPOConfig,
    AutoModelForSeq2SeqLMWithValueHead,
    create_reference_model
)

from libs.utils.logging import add_color_formater
from libs.data_helpers.bytedataset import ByteDataset
from libs.utils.seeding import seed_everything
from rlhf.arguments import (
    add_training_arguments,
    add_tokenizer_arguments,
    add_model_arguments,
    add_data_arguments,
    add_other_arguments
)
from rlhf.mapping import resolve_tokenizer_class
from rlhf.evaluation import evaluate_generator
from rlhf.reward import Rouge1F1Reward, SentenceEmbeddingSimilarityReward

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
add_color_formater(logging.root)


def get_xsum_args():
    from rlhf.config import xsum_setting
    args = argparse.Namespace()
    xsum_setting(args)
    return args


def get_vietnews_args():
    from rlhf.config import vietnews_setting
    args = argparse.Namespace()
    vietnews_setting(args)
    return args


def get_data_specific_args(data_name):
    if data_name == "xsum":
        return get_xsum_args()
    elif data_name == "vietnews":
        return get_vietnews_args()
    else:
        raise Exception("Data '{}' is not supported.".format(data_name))


def get_metric(valid_stats, metric_for_best_model):
    if metric_for_best_model not in valid_stats:
        avail_stats = list(valid_stats.keys())
        avail_stats = ", ".join(valid_stats)
        raise Exception("Metric {} is not in the valid stats. Available stat are: {}.".format(avail_stats))
    return valid_stats[metric_for_best_model]


def get_data_collator(
    tokenizer,
    input_name,
    output_name,
    max_input_len: int = 512,
    max_output_len: int = 80,
):
    def collate_fn(items: List[Dict[Text, Text]]):
        documents = []
        summaries = []
        ids = []
        for item in items:
            documents.append(item[input_name])
            summaries.append(item[output_name])
            ids.append(item.get("id", None))
        
        encoded_documents = tokenizer(
            documents,
            padding=False,
            truncation=True,
            max_length=max_input_len,
        )
        encoded_summaries = tokenizer(
            summaries,
            padding=False,
            truncation=True,
            max_length=max_output_len
        )

        return {
            "ids": ids,
            "input_ids": [torch.tensor(v) for v in encoded_documents.input_ids],
            "decoder_input_ids": [torch.tensor(v) for v in encoded_summaries.input_ids],
            input_name: documents,
            output_name: summaries
        }
    return collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name",
                        choices=["xsum", "vietnews"],
                        default="xsum")
    add_training_arguments(parser)
    add_tokenizer_arguments(parser)
    add_model_arguments(parser)
    add_data_arguments(parser)
    add_other_arguments(parser)
    args = parser.parse_args()

    seed_everything(args.seed)
    data_specific_args = get_data_specific_args(args.data_name)
    cfg = deepcopy(args.__dict__)
    cfg.update(**deepcopy(data_specific_args.__dict__))
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)
    with open(os.path.join(args.model_save_path, "training_config.json"), "w") as writer:
        json.dump(cfg, writer, indent=4, ensure_ascii=False)

    # tokenizer
    tokenizer_class = resolve_tokenizer_class(args.tokenizer_class)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path)

    # policy
    policy = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(args.model_path)
    sft_model = create_reference_model(policy)

    # initialize trainer
    ppo_config = PPOConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        log_with="tensorboard",
        accelerator_kwargs={
            "logging_dir": args.logging_dir
        }
    )
    ppo_trainer = PPOTrainer(ppo_config, policy, sft_model, tokenizer)
    device = ppo_trainer.accelerator.device
    
    # data loader
    dataset = ByteDataset(data_path=args.data_path, idx_record_size=6)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=get_data_collator(
            tokenizer,
            args.input_name,
            args.output_name,
            max_input_len=data_specific_args.total_len,
            max_output_len=data_specific_args.max_len
        )
    )
    valid_dataloader = None
    if args.do_eval:
        valid_dataset = ByteDataset(data_path=args.valid_data_path, idx_record_size=6)
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=get_data_collator(
                tokenizer,
                args.input_name,
                args.output_name,
                max_input_len=data_specific_args.total_len,
                max_output_len=data_specific_args.max_len
            )
        )

    # generation config
    generation_kwargs = {
        "min_length": 4,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": data_specific_args.max_len,
        "no_repeat_ngram_size": 3
    }

    eval_generation_kwargs = {
        "no_repeat_ngram_size": 3,
        "max_new_tokens": data_specific_args.max_len,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    # reward model
    if args.reward_model == "rouge1-f1":
        reward_model = Rouge1F1Reward()
    elif args.reward_model == "vector_similarity":
        reward_model = SentenceEmbeddingSimilarityReward(sim_model=args.sim_model)
        # we need to seed the vector-similarity reward model
        if args.baseline == "avg":
            sample_dataset = ByteDataset(data_path=args.valid_data_path, idx_record_size=6)
            sample_dataloader = DataLoader(
                sample_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=get_data_collator(
                    tokenizer,
                    args.input_name,
                    args.output_name,
                    max_input_len=data_specific_args.total_len,
                    max_output_len=data_specific_args.max_len
                )
            )
            logger.info("Iterating sample dataset to seed the reward model...")
            progress_bar = tqdm(
                total=min(len(sample_dataloader), args.vector_sim_data_cut),
                desc="Batch")
            for idx, batch in enumerate(sample_dataloader):
                ref_summary = batch[args.output_name][0]
                input_ids = batch["input_ids"][0].to(device)
                hyp_summary_output = ppo_trainer.generate(input_ids, **eval_generation_kwargs)
                hyp_summary_output = hyp_summary_output.squeeze()
                hyp_summary = tokenizer.decode(
                        hyp_summary_output, clean_up_tokenization_spaces=False, skip_special_tokens=True)
                reward_model.cal_reward(hyp_summary, ref_summary)
                progress_bar.update(1)
                if (idx + 1) == args.vector_sim_data_cut:
                    break
    else:
        raise Exception("Reward model '{}' is not supported.".format(args.reward_model))
    
    # pre-training evaluation
    if args.do_eval and args.eval_on_first_step:
        valid_stats = evaluate_generator(
            ppo_trainer.accelerator.unwrap_model(ppo_trainer.model).pretrained_model,
            valid_dataloader,
            tokenizer,
            device,
            eval_generation_kwargs,
            input_name=args.input_name,
            output_name=args.output_name
        )
        ppo_trainer.accelerator.log(valid_stats, step=0)

    best_metric = float("-inf")
    best_checkpoint = None
    global_step = 0
    total_steps = len(dataloader) * args.num_train_epochs
    progress_bar = tqdm(desc="Step", total=total_steps)
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(dataloader):
            summaries = batch[args.output_name]
            query_tensors = batch["input_ids"]
            query_tensors = [q.to(device) for q in query_tensors]
            response_tensors = []
            rewards = []
            ignored_idxs = set() # PPO raise ValueError when repsonse tensor is less then 4 tokens
            for i, query in enumerate(query_tensors):
                # calculate response tensor
                response = ppo_trainer.generate(query, **generation_kwargs)
                response = response.squeeze()
                if len(response) < 4:
                    ignored_idxs.add(i)
                    continue
                response_tensors.append(response)
                response_text = tokenizer.decode(
                    response, clean_up_tokenization_spaces=False, skip_special_tokens=True)

                # calculate reward
                summary = summaries[i]
                if args.baseline == "avg":
                    avg_reward = reward_model.get_avg_reward()
                else:
                    avg_reward = 0
                reward = reward_model.cal_reward(response_text, summary) - avg_reward
                rewards.append(torch.tensor(reward).to(device))

            # filter query_tensors
            query_tensors = [query_tensors[i] for i in range(len(query_tensors)) if i not in ignored_idxs]
            if not query_tensors:
                continue

            # PPO step
            train_stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(train_stats, batch, rewards)

            # Save model
            if (global_step + 1) % args.save_steps == 0:
                # evaluation
                valid_stats = evaluate_generator(
                    ppo_trainer.accelerator.unwrap_model(ppo_trainer.model).pretrained_model,
                    valid_dataloader,
                    tokenizer,
                    device,
                    generation_kwargs,
                    input_name=args.input_name,
                    output_name=args.output_name
                )
                ppo_trainer.accelerator.log(valid_stats, step=global_step + 1)

                # save checkpoint
                if ppo_trainer.accelerator.is_main_process:
                    cp_name = f"checkpoint-{global_step + 1}"
                    cp_path = os.path.join(args.model_save_path, cp_name)
                    if not os.path.exists(cp_path):
                        os.makedirs(cp_path)
                    metric = get_metric(valid_stats, args.metric_for_best_model)
                    if not args.greater_is_better:
                        metric = -metric
                    if metric > best_metric:
                        best_metric = metric
                        best_checkpoint = cp_name
                        logger.info("New best checkpoint: '{}'".format(cp_name))
                    ppo_trainer.tokenizer.save_pretrained(cp_path)
                    ppo_trainer.accelerator.unwrap_model(ppo_trainer.model).save_pretrained(cp_path)

                    # remove old checkpoints if neccessary
                    all_checkpoints = os.listdir(args.model_save_path)
                    all_checkpoints = [cp for cp in all_checkpoints if cp != "training_config.json"]
                    all_checkpoints = [cp for cp in all_checkpoints if cp != best_checkpoint]
                    all_checkpoints = [os.path.join(args.model_save_path, cp) for cp in all_checkpoints]
                    all_checkpoints = sorted(all_checkpoints, key=lambda x: os.path.getctime(x), reverse=True)
                    all_checkpoints = [
                        os.path.join(args.model_save_path, best_checkpoint)
                    ] + all_checkpoints
                    tobe_removed_checkpoints = all_checkpoints[args.keep_checkpoint_max:]
                    for cp in tobe_removed_checkpoints:
                        logger.info("Deleting {} since maximum kept checkpoints is {}...".format(cp, args.keep_checkpoint_max))
                        shutil.rmtree(cp)
            progress_bar.update(1)
            global_step += 1


if __name__ == "__main__":
    main()
