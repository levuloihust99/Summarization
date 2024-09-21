import os
import logging
import argparse

from trl import AutoModelForSeq2SeqLMWithValueHead
from transformers.modeling_utils import _load_state_dict_into_model

from libs.utils.logging import do_setup_logging
from rlhf.nn.checkpoint import load_checkpoint_state

do_setup_logging(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", default="VietAI/vit5-base-vietnews-summarization")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output_path", default="t5seq2seq_converted")
    args = parser.parse_args()

    policy = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(args.pretrained_path)
    checkpoint_state = load_checkpoint_state(args.ckpt_path)
    policy.v_head.load_state_dict(checkpoint_state.model_state.v_head)
    _load_state_dict_into_model(
        policy.pretrained_model,
        checkpoint_state.model_state.pretrained_model,
        start_prefix=""
    )
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    policy.pretrained_model.save_pretrained(args.output_path)


if __name__ == "__main__":
    main()
