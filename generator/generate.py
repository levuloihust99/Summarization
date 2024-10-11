import os
import json
import random
import logging
import argparse

from tqdm import tqdm
from typing import Text, Dict, List, Any

from .builtin_generators import resolve_summarizer_class
from .samplers import noninfluent_sampler, top10_30_sampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(data_path: Text) -> List[Dict[Text, Any]]:
    data = []
    with open(data_path, "r") as reader:
        for line in reader:
            data.append(json.loads(line.strip()))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-i", default="data/vietnews-hf/ranking-data/20ksubset/data.jsonl")
    parser.add_argument("--output_path", "-o", default="data/vietnews-hf/ranking-data/20ksubset-l1/VietAI_vit5.jsonl")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--block_n_grams", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=150)
    parser.add_argument("--type", default="t5_cond")
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--max_input_len", type=int, default=2048)
    parser.add_argument("--model_path", default="VietAI/vit5-base-vietnews-summarization")
    parser.add_argument("--input_key", default="input")
    parser.add_argument("--output_key", default="output")
    parser.add_argument("--id_key", default="sampleId")
    parser.add_argument("--sampler", default=None, choices=["top10_30_sampler", "noninfluent_sampler"])
    args = parser.parse_args()

    data = load_data(args.input_path)
    model_class = resolve_summarizer_class(args.type)
    summarizer = model_class(args.model_path, args.tokenizer_path, args.max_input_len)

    kwargs = {}
    if args.sampler:
        kwargs["with_sampling"] = True
        kwargs["sampler"] = eval(args.sampler)

    out_data = []
    batch = []
    batch_size = args.batch_size
    progress_bar = tqdm(total=len(data), desc="Processing")

    processed = set()
    if os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as reader:
            for line in reader:
                item = json.loads(line.strip())
                processed.add(item[args.id_key])

    with open(args.output_path, "a", encoding="utf-8") as writer:
        for item in data:
            if item[args.id_key] in processed:
                progress_bar.update(1)
                continue
            batch.append(item)
            if len(batch) == batch_size:
                if args.sampler == "top10_30_sampler":
                    range_lower = max(50, args.max_length - 100)
                    range_upper = max(args.max_length, range_lower + 50)
                    selected_max_len = random.randint(range_lower, range_upper)
                    kwargs.update(max_length=selected_max_len)
                else:
                    kwargs.update(max_length=args.max_length)
                outputs = summarizer.greedy(
                    [_item[args.input_key] for _item in batch],
                    block_n_grams=args.block_n_grams,
                    **kwargs
                )
                done_batch = []
                for output, _item in zip(outputs, batch):
                    done_batch.append({
                        args.id_key: _item[args.id_key],
                        args.input_key: _item[args.input_key],
                        args.output_key: output
                    })
                    progress_bar.update(1)
                out_data.extend(done_batch)
                for done_item in done_batch:
                    writer.write(json.dumps(done_item, ensure_ascii=False) + "\n")
                    writer.flush()
                batch = []

        if len(batch) > 0:
            logger.info("Last batch...")
            if args.sampler == "top10_30_sampler":
                range_lower = max(50, args.max_length - 100)
                range_upper = max(args.max_length, range_lower + 50)
                selected_max_len = random.randint(range_lower, range_upper)
                kwargs.update(max_length=selected_max_len)
            outputs = summarizer.greedy(
                [_item[args.input_key] for _item in batch],
                block_n_grams=args.block_n_grams,
                **kwargs
            )
            done_batch = []
            for output, _item in zip(outputs, batch):
                out_data.append({
                    args.id_key: _item[args.id_key],
                    args.input_key: _item[args.input_key],
                    args.output_key: output
                })
                progress_bar.update(1)
            out_data.extend(done_batch)
            for done_item in done_batch:
                writer.write(json.dumps(done_item, ensure_ascii=False) + "\n")
                writer.flush()
            batch = []


if __name__ == "__main__":
    main()
