# Training RLHF

## Prepare training config
```python
import json

config = { # this is an example config
    "data_name": "vietnews",
    "train_data_path": "data/vietnews-hf/full/bytedataset/train",
    "eval_data_path": "data/vietnews-hf/full/bytedataset/eval",
    "model_save_path": "assets/vietnews-hf/outputs/rlhf",
    "tokenizer_path": "VietAI/vit5-base-vietnews-summarization",
    "tokenizer_class": "AutoTokenizer",
    "model_path": "VietAI/vit5-base-vietnews-summarization",
    "model_class": "AutoModelForSeq2SeqLM",
    "reward_model": "crossenc",
    "baseline": "zero",
    "sim_model": "NtDNlp/sentence-embedding-vietnamese",
    "crossenc_ckpt_path": "checkpoint-003100.pth",
    "crossenc_pretrained": "VietAI/vit5-base",
    "crossenc_sep_token": "<extra_id_0>",
    "batch_size": 8,
    "save_steps": 100,
    "num_train_epochs": 50,
    "logging_steps": 10,
    "do_eval": True,
    "logging_dir": "assets/vietnews-hf/logs/rlhf",
    "learning_rate": 1e-6,
    "seed": 12345,
    "metric_for_best_model": "eval/reward",
    "greater_is_better": True,
    "input_name": "article",
    "output_name": "abstract",
    "anchor": "input",
    "keep_checkpoint_max": 5,
    "eval_mode": "reward"
}

with open("training_config.json", "w") as writer:
    json.dump(config, writer, indent=4, ensure_ascii=False)
```

## Run train
```bash
python -m rlhf.train --hparams training_config.json
```
