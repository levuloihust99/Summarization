# tokenizer
tokenizer_path = None

# model
model_path = None

# dataset
train_data_path = None
valid_data_path = None

# training config
output_dir = "assets/outputs"
do_train = True
do_eval = True
learning_rate = 4e-3
num_train_epochs = 10
warmup_ratio = 0.0
warmup_steps = 0
weight_decay = 0
per_device_train_batch_size = 2
per_device_eval_batch_size = 2
logging_dir = "assets/logs"
group_by_length = False
save_strategy = "steps"
save_steps = 100
evaluation_strategy = "steps"
eval_steps = 100
save_total_limit = 2
fp16 = False
gradient_accumulation_steps = 1
gradient_checkpointing = False
log_level = "info"
logging_steps = 50
logging_first_step = True
max_grad_norm = 2.0
label_smoothing_factor = 0.1
report_to = ["tensorboard"]
load_best_model_at_end = True
metric_for_best_model = "acc"
greater_is_better = True
predict_with_generate = False
resume_from_checkpoint = None
remove_unused_columns = False
generation_max_length = 100
generation_num_beams = 1
input_name = "prompt"
output_name = "completion"
data_seed = None
max_input_len = None
max_output_len = None

# data config
input_transform = None # [json_sequentialize]
output_transform = None # [json_sequentialize]


def to_json():
    config = {
        "tokenizer_path": tokenizer_path,
        "model_path": model_path,
        "train_data_path": train_data_path,
        "valid_data_path": valid_data_path,
        "output_dir": output_dir,
        "do_train": do_train,
        "do_eval": do_eval,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "logging_dir": logging_dir,
        "group_by_length": group_by_length,
        "save_strategy": save_strategy,
        "save_steps": save_steps,
        "evaluation_strategy": evaluation_strategy,
        "eval_steps": eval_steps,
        "save_total_limit": save_total_limit,
        "fp16": fp16,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_checkpointing": gradient_checkpointing,
        "log_level": log_level,
        "logging_steps": logging_steps,
        "logging_first_step": logging_first_step,
        "max_grad_norm": max_grad_norm,
        "label_smoothing_factor": label_smoothing_factor,
        "report_to": report_to,
        "load_best_model_at_end": load_best_model_at_end,
        "metric_for_best_model": metric_for_best_model,
        "greater_is_better": greater_is_better,
        "predict_with_generate": predict_with_generate,
        "resume_from_checkpoint": resume_from_checkpoint,
        "generation_max_length": generation_max_length,
        "generation_num_beams": generation_num_beams,
        "input_name": input_name,
        "output_name": output_name,
        "data_seed": data_seed,
        "max_input_len": max_input_len,
        "max_output_len": max_output_len,
        "input_transform": input_transform,
        "output_transform": output_transform
    }
    return config
