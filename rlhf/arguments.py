import argparse


def add_training_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--model_save_path", default="outputs")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--do_eval", action="store_true", default=False)
    parser.add_argument("--logging_dir", default="logs")
    parser.add_argument("--keep_checkpoint_max", type=int, default=3)
    parser.add_argument("--eval_on_first_step", action="store_true", default=False)


def add_tokenizer_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--tokenizer_path", default="Yale-LILY/brio-xsum-cased")
    parser.add_argument("--tokenizer_class", default="PegasusTokenizer")


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--model_path", default="Yale-LILY/brio-xsum-cased")
    parser.add_argument("--model_class", default="PegasusForConditionalGeneration")


def add_data_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--data_path", default="data/xsum/full/bytedataset/train")
    parser.add_argument("--valid_data_path", default="")


def add_other_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--seed", type=int, default=12345)
