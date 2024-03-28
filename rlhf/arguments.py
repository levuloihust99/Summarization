import argparse


def add_training_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--model_save_path")
    parser.add_argument("--save_steps", type=int)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--logging_dir")
    parser.add_argument("--log_level", choices=["info", "debug", "warning", "error", "critical"])
    parser.add_argument("--log_with", type=eval)
    parser.add_argument("--logging_steps", type=int)
    parser.add_argument("--wandb_api_key")
    parser.add_argument("--keep_checkpoint_max", type=int)
    parser.add_argument("--eval_on_first_step", action="store_true")
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--resume_from_checkpoint")
    parser.add_argument("--eval_mode", choices=["reward", "metric", "all"])


def add_tokenizer_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--tokenizer_path")
    parser.add_argument("--tokenizer_class")


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--model_path")
    parser.add_argument("--model_class")


def add_data_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--data_name", choices=["xsum", "vietnews", "cnndm"])
    parser.add_argument("--train_data_path")
    parser.add_argument("--eval_data_path")


def add_other_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--seed", type=int)
    parser.add_argument("--reward_model", choices=["rouge1-f1", "vector_similarity", "crossenc"])
    parser.add_argument("--greater_is_better", type=eval)
    parser.add_argument("--metric_for_best_model")
    parser.add_argument("--baseline", choices=["zero", "avg"])
    parser.add_argument("--sim_model")
    parser.add_argument("--crossenc_ckpt_path")
    parser.add_argument("--crossenc_sep_token")
    parser.add_argument("--crossenc_pretrained")
    parser.add_argument("--input_name")
    parser.add_argument("--output_name")
    parser.add_argument("--seed_reward_data_cut", type=int)
    parser.add_argument("--anchor", choices=["input", "output"])
    parser.add_argument("--hparams", default="{}")


def create_parser():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    add_training_arguments(parser)
    add_tokenizer_arguments(parser)
    add_model_arguments(parser)
    add_data_arguments(parser)
    add_other_arguments(parser)
    return parser
