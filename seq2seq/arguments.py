import argparse


def create_parser():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--tokenizer_path", required=True,
                        help="Path to the tokenizer to be instantiated by AutoTokenizer")
    parser.add_argument("--model_path", required=True,
                        help="Path to the model to be instantiated by AutoModelForSeq2SeqLM")
    parser.add_argument("--train_data_path",
                        help="Path to the train bytedataset directory.")
    parser.add_argument("--valid_data_path",
                        help="Path to the validation bytedataset directory.")
    parser.add_argument("--output_dir",
                        help="Path to the output directory, which stores checkpoints.")
    parser.add_argument("--do_train", type=eval,
                        help="Whether to perform training.")
    parser.add_argument("--do_eval", type=eval,
                        help="Whether to perform evaluation.")
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--warmup_ratio", type=float)
    parser.add_argument("--warmup_steps", type=int)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--per_device_train_batch_size", type=int)
    parser.add_argument("--per_device_eval_batch_size", type=int)
    parser.add_argument("--logging_dir",
                        help="Path to the logging directory, may be tensorboard logging.")
    parser.add_argument("--group_by_length", type=eval)
    parser.add_argument("--save_strategy")
    parser.add_argument("--save_steps", type=int)
    parser.add_argument("--evaluation_strategy")
    parser.add_argument("--eval_steps", type=int)
    parser.add_argument("--save_total_limit", type=int)
    parser.add_argument("--fp16", type=eval)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--gradient_checkpointing", type=eval)
    parser.add_argument("--log_level")
    parser.add_argument("--logging_steps", type=int)
    parser.add_argument("--logging_first_step", type=eval)
    parser.add_argument("--max_grad_norm", type=float)
    parser.add_argument("--label_smoothing_factor", type=float)
    parser.add_argument("--load_best_model_at_end", type=eval)
    parser.add_argument("--metric_for_best_model")
    parser.add_argument("--greater_is_better", type=eval)
    parser.add_argument("--predict_with_generate", type=eval)
    parser.add_argument("--resume_from_checkpoint")
    parser.add_argument("--remove_unused_columns", type=eval)
    parser.add_argument("--generation_max_length", type=int)
    parser.add_argument("--generation_num_beams", type=int)
    parser.add_argument("--input_name")
    parser.add_argument("--output_name")
    parser.add_argument("--data_seed", type=int)
    parser.add_argument("--max_input_len", type=int)
    parser.add_argument("--max_output_len", type=int)

    # data params
    parser.add_argument("--input_transform", choices=["json_sequentialize"])
    parser.add_argument("--output_transform", choices=["json_sequentialize"])

    return parser
