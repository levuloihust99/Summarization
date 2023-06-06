import re
import os
import json
import string
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from typing import Text, List

from .config import default_config
from .arguments import create_parser
from libs.data_helpers.bytedataset import ByteDataset
from .preprocessing import NFKCNormalizer
from .dataloader import get_collate_fn
from libs.utils.rouge_calculator import _rouge_n_sentence_level, _f_score


def calculate_rouge_score(ref: Text, pred: Text, ns: List[int] = [1]):
    ref = ref.lower()
    pred = pred.lower()

    punc_patt = re.compile(f"[{re.escape(string.punctuation)}]")
    ref = punc_patt.sub(" ", ref)
    ref_tokens = ref.split()
    pred = punc_patt.sub(" ", pred)
    pred_tokens = pred.split()
    
    score = {}
    for n in ns:
        score[n] = _rouge_n_sentence_level(pred_tokens, ref_tokens, n).to_score(alpha=0.5)
    return score


def get_compute_metrics(tokenizer):
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        labels[labels == -100] = tokenizer.pad_token_id
        preds = logits.argmax(axis=-1)
        
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
        for ref, pred in zip(references, predictions):
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

        active_mask = (labels != tokenizer.pad_token_id)
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
            "ppl": float(ppl),
            "rouge-1-p": r1_p,
            "rouge-1-r": r1_r,
            "rouge-1-f1": r1_f1,
            "rouge-2-p": r2_p,
            "rouge-2-r": r2_r,
            "rouge-2-f1": r2_f1
        }
    return compute_metrics


def get_config(cfg, args):
    for k, v in args.__dict__.items():
        cfg.__dict__[k] = v
    return cfg


def main():
    parser = create_parser()
    args = parser.parse_args()
    cfg = get_config(default_config, args)

    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    with open(os.path.join(cfg.output_dir, "training_config.json"), "w") as writer:
        json.dump(cfg.to_json(), writer, indent=4, ensure_ascii=False)

    normalizer = NFKCNormalizer()
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_path)

    # train dataloader
    train_dataset = None
    if cfg.do_train:
        train_dataset = ByteDataset(cfg.train_data_path, 6)
    valid_dataset = None
    if cfg.do_eval:
        valid_dataset = ByteDataset(cfg.valid_data_path, 6)
    data_collate_fn = get_collate_fn(
        tokenizer=tokenizer,
        normalizer=normalizer,
        decoder_start_token_id=model.config.decoder_start_token_id,
        input_transform=cfg.input_transform,
        output_transform=cfg.output_transform,
        max_input_len=cfg.max_input_len,
        max_output_len=cfg.max_output_len,
        input_name=cfg.input_name,
        output_name=cfg.output_name
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
        data_seed=cfg.data_seed
    )

    # Trainer: trainer takes care of switching between train/eval mode
    # no need to do model.train() or model.eval() manually
    compute_metrics = get_compute_metrics(tokenizer)
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
