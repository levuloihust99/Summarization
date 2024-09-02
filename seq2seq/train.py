import re
import os
import json
import copy
import wandb
import string
import random
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from typing import Text, List

from .configuration import Seq2SeqConfig
from .arguments import create_parser
from .preprocessing import NFKCNormalizer
from .dataloader import get_collate_fn
from .modeling import init_model
from libs.data_helpers.bytedataset import ByteDataset
from libs.utils.rouge_calculator import _rouge_n_sentence_level, _f_score


def override_defaults(hparams, args):
    for key in args:
        hparams[key] = args[key]
    return hparams


def calculate_rouge_score(pred: Text, ref: Text, ns: List[int] = [1]):
    pred = pred.lower()
    ref = ref.lower()

    punc_patt = re.compile(f"[{re.escape(string.punctuation)}]")
    pred = punc_patt.sub(" ", pred)
    pred_tokens = pred.split()
    ref = punc_patt.sub(" ", ref)
    ref_tokens = ref.split()
    
    score = {}
    for n in ns:
        score[n] = _rouge_n_sentence_level(pred_tokens, ref_tokens, n).to_score(alpha=0.5)
    return score


def get_compute_metrics(tokenizer, predict_with_generate: bool):
    def compute_metrics(eval_preds):
        if predict_with_generate:
            preds, labels = eval_preds
            preds[preds == -100] = tokenizer.pad_token_id
            labels[labels == -100] = tokenizer.pad_token_id

            references = []
            references = [
                tokenizer.decode(tokens, clean_up_tokenization_spaces=False, skip_special_tokens=True)
                for tokens in labels
            ]
            predictions = [
                tokenizer.decode(tokens, clean_up_tokenization_spaces=False, skip_special_tokens=True)
                for tokens in preds
            ]

            scores = []
            for pred, ref in zip(predictions, references):
                score = calculate_rouge_score(pred, ref, [1, 2])
                scores.append(score)
            
            r1_precisions = []
            r1_recalls = []
            r2_precisions = []
            r2_recalls = []

            for score in scores:
                r1_precisions.append(score[1]["p"])
                r1_recalls.append(score[1]["r"])
                r2_precisions.append(score[2]["p"])
                r2_recalls.append(score[2]["r"])
            
            r1_p = sum(r1_precisions) / len(r1_precisions)
            r1_r = sum(r1_recalls) / len(r1_recalls)
            r1_f1 = _f_score(r1_p, r1_r, alpha=0.5)
            r2_p = sum(r2_precisions) / len(r2_precisions)
            r2_r = sum(r2_recalls) / len(r2_recalls)
            r2_f1 = _f_score(r2_p, r2_r, alpha=0.5)

            return {
                "rouge-1-p": r1_p,
                "rouge-1-r": r1_r,
                "rouge-1-f1": r1_f1,
                "rouge-2-p": r2_p,
                "rouge-2-r": r2_r,
                "rouge-2-f1": r2_f1
            }

        # else, predict without generate
        logits, labels = eval_preds
        active_mask = (labels != -100)
        num_active_tokens = active_mask.sum()
        flatten_mask = active_mask.reshape(-1)
        bsz, seq_len = labels.shape

        flatten_logits = logits.reshape(bsz * seq_len, -1)
        flatten_labels = labels.reshape(bsz * seq_len)

        active_logits = flatten_logits[flatten_mask]
        active_logprobs = -torch.nn.functional.log_softmax(
            torch.tensor(active_logits), dim=-1).numpy()
        active_labels = flatten_labels[flatten_mask]
        active_preds  = active_logits.argmax(axis=-1)

        acc = (active_preds == active_labels).sum() / num_active_tokens

        active_xent = np.take(active_logprobs, active_labels).mean()
        ppl = np.exp(min(active_xent, np.log(100000)))

        return {
            "acc": float(acc),
            "ppl": float(ppl)
        }
    return compute_metrics


def main():
    parser = create_parser()
    args = parser.parse_args()
    args_json = copy.deepcopy(args.__dict__)
    hparams = args_json.pop('hparams')
    if args.hparams.endswith('.json'):
        with open(args.hparams, "r") as f:
            hparams = json.load(f)
    else:
        hparams = json.loads(args.hparams)
    hparams = override_defaults(hparams, args_json)
    cfg = Seq2SeqConfig(**hparams)

    run_id = None
    if cfg.add_run_id:
        run_id = "".join(
            random.choice(string.digits + string.ascii_uppercase) for _ in range(16)
        )
    if run_id:
        cfg.output_dir = os.path.join(cfg.output_dir, run_id)
        cfg.logging_dir = os.path.join(cfg.logging_dir, run_id)

    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    with open(os.path.join(cfg.output_dir, "training_config.json"), "w") as writer:
        json.dump(cfg.to_json(), writer, indent=4, ensure_ascii=False)

    normalizer = NFKCNormalizer()
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path, use_fast=cfg.use_fast)
    if cfg.model_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_path)
    else:
        model = init_model(
            cfg.model_type,
            decoder_start_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            vocab_size=tokenizer.vocab_size
        )

    # train dataloader
    train_dataset = None
    if cfg.do_train:
        train_dataset = ByteDataset(cfg.train_data_path, 6)
    valid_dataset = None
    if cfg.do_eval:
        valid_dataset = ByteDataset(cfg.valid_data_path, 6)
    decoder_start_token_id = (
        model.config.decoder_start_token_id or
        tokenizer.bos_token_id or
        tokenizer.pad_token_id
    )
    data_collate_fn = get_collate_fn(
        tokenizer=tokenizer,
        normalizer=normalizer,
        decoder_start_token_id=decoder_start_token_id,
        input_transform=cfg.input_transform,
        output_transform=cfg.output_transform,
        max_input_len=cfg.max_input_len,
        max_output_len=cfg.max_output_len,
        input_name=cfg.input_name,
        output_name=cfg.output_name
    )
    
    # wandb setup
    if "wandb" in cfg.report_to:
        wandb.login(key=cfg.wandb_api_key)
        wandb.init(
            project="Seq2Seq",
            name=run_id or "Default",
            config=cfg.to_json()
        )

    # training arguments
    training_args = Seq2SeqTrainingArguments(
        cfg.output_dir,
        do_train=cfg.do_train,
        do_eval=cfg.do_eval,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        logging_dir=cfg.logging_dir,
        group_by_length=cfg.group_by_length,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        evaluation_strategy=cfg.evaluation_strategy,
        eval_steps=cfg.eval_steps,
        save_total_limit=cfg.save_total_limit,
        fp16=cfg.fp16,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        log_level=cfg.log_level,
        logging_steps=cfg.logging_steps,
        logging_first_step=cfg.logging_first_step,
        max_grad_norm=cfg.max_grad_norm,
        label_smoothing_factor=cfg.label_smoothing_factor,
        report_to=cfg.report_to,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        predict_with_generate=cfg.predict_with_generate,
        remove_unused_columns=cfg.remove_unused_columns,
        generation_max_length=cfg.generation_max_length,
        generation_num_beams=cfg.generation_num_beams,
        data_seed=cfg.data_seed,
    )

    # Trainer: trainer takes care of switching between train/eval mode.
    # No need to do model.train() or model.eval() manually
    compute_metrics = get_compute_metrics(tokenizer, cfg.predict_with_generate)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collate_fn,
        compute_metrics=compute_metrics
    )
    trainer.train(
        cfg.resume_from_checkpoint,
        ignore_keys_for_eval=["past_key_values", "encoder_last_hidden_state"]
    )


if __name__ == "__main__":
    main()
