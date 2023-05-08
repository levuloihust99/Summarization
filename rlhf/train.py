import os
import torch
import shutil
import logging
import argparse
from tqdm import tqdm
from typing import List, Dict, Text, Any

from torch.utils.data import DataLoader
from transformers import BertTokenizer
from trl import (
    PPOTrainer,
    PPOConfig,
    AutoModelForSeq2SeqLMWithValueHead,
    create_reference_model
)

from libs.utils.logging import add_color_formater
from libs.data_helpers.bytedataset import ByteDataset
from libs.utils.seeding import seed_everything
from libs.utils.rouge_calculator import _rouge_n_sentence_level
from rlhf.arguments import (
    add_training_arguments,
    add_tokenizer_arguments,
    add_model_arguments,
    add_data_arguments,
    add_other_arguments
)
from rlhf.mapping import resolve_tokenizer_class
from rlhf.evaluation import evaluate_generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
add_color_formater(logging.root)


def get_xsum_args():
    from rlhf.config import xsum_setting
    args = argparse.Namespace()
    xsum_setting(args)
    return args


def get_data_specific_args(model_name):
    if model_name == "brio-xsum":
        return get_xsum_args()
    else:
        raise Exception("Model name '{}' is not supported.".format(model_name))


def get_data_collator(
    tokenizer,
    max_input_len: int = 512,
    max_output_len: int = 80,
):
    def collate_fn(items: List[Dict[Text, Text]]):
        documents = []
        summaries = []
        ids = []
        for item in items:
            documents.append(item["document"])
            summaries.append(item["summary"])
            ids.append(item["id"])
        
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
            "document": documents,
            "summary": summaries
        }
    return collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", choices=["brio-xsum"], default="brio-xsum")
    add_training_arguments(parser)
    add_tokenizer_arguments(parser)
    add_model_arguments(parser)
    add_data_arguments(parser)
    add_other_arguments(parser)
    args = parser.parse_args()

    seed_everything(args.seed)
    data_specific_args = get_data_specific_args(args.model_name)

    # tokenizer
    tokenizer_class = resolve_tokenizer_class(args.tokenizer_class)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path)
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    basic_tokenizer = bert_tokenizer.basic_tokenizer

    # policy
    policy = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(args.model_path)
    sft_model = create_reference_model(policy)

    # initialize trainer
    ppo_config = PPOConfig(
        batch_size=args.batch_size,
        log_with="tensorboard",
        accelerator_kwargs={
            "logging_dir": args.logging_dir
        }
    )
    ppo_trainer = PPOTrainer(ppo_config, policy, sft_model, tokenizer)
    
    # data loader
    dataset = ByteDataset(data_path=args.data_path, idx_record_size=6)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=get_data_collator(
            tokenizer,
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
                max_input_len=data_specific_args.total_len,
                max_output_len=data_specific_args.max_len
            )
        )

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": data_specific_args.max_len,
        "no_repeat_ngram_size": 3
    }

    # pre-training evaluation
    if args.do_eval and args.eval_on_first_step:
        valid_stats = evaluate_generator(
            ppo_trainer.accelerator.unwrap_model(ppo_trainer.model).pretrained_model,
            valid_dataloader,
            tokenizer,
            generation_kwargs
        )
        ppo_trainer.accelerator.log(valid_stats, step=0)

    best_rouge1_f1 = float("-inf")
    best_checkpoint = None
    for step, batch in enumerate(tqdm(dataloader)):
        documents = batch["document"]
        summaries = batch["summary"]
        query_tensors = batch["input_ids"]
        response_tensors = []
        rewards = []
        for i, query in enumerate(query_tensors):
            # calculate response tensor
            response = ppo_trainer.generate(query, **generation_kwargs)
            response = response.squeeze()
            response_tensors.append(response)
            response_text = tokenizer.decode(
                response, clean_up_tokenization_spaces=False, skip_special_tokens=True)
            response_tokens = basic_tokenizer.tokenize(response_text)

            # calculate reward (ROUGE-1 F1 score)
            summary = summaries[i]
            summary_tokens = basic_tokenizer.tokenize(summary)
            metric = _rouge_n_sentence_level(response_tokens, summary_tokens, 1)
            reward = metric.to_score(alpha=0.5)["f"] * 10
            rewards.append(torch.tensor(reward))

        # PPO step
        train_stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(train_stats, batch, rewards)

        # Save model
        if (step + 1) % args.save_steps == 0:
            # evaluation
            valid_stats = evaluate_generator(
                ppo_trainer.accelerator.unwrap_model(ppo_trainer.model).pretrained_model,
                valid_dataloader,
                tokenizer,
                generation_kwargs
            )
            ppo_trainer.accelerator.log(valid_stats, step=step + 1)

            # save checkpoint
            if ppo_trainer.accelerator.is_main_process:
                cp_name = f"checkpoint-{step + 1}"
                cp_path = os.path.join(args.model_save_path, cp_name)
                if not os.path.exists(cp_path):
                    os.makedirs(cp_path)
                rouge1_f1 = valid_stats["eval/rouge1-f1"]
                if rouge1_f1 > best_rouge1_f1:
                    best_rouge1_f1 = rouge1_f1
                    best_checkpoint = cp_name
                ppo_trainer.tokenizer.save_pretrained(cp_path)
                ppo_trainer.accelerator.unwrap_model(ppo_trainer.model).save_pretrained(cp_path)

                # remove old checkpoints if neccessary
                all_checkpoints = os.listdir(args.model_save_path)
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


if __name__ == "__main__":
    main()
