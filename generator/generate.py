import os
import json
import logging
import argparse

from tqdm import tqdm
from typing import Text, Dict, List, Any

from .builtin_generators import resolve_summarizer_class

logging.basicConfig(level=logging.INFO)


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
    parser.add_argument("--model_path", default="VietAI/vit5-base-vietnews-summarization")
    parser.add_argument("--input_key", default="input")
    parser.add_argument("--output_key", default="output")
    parser.add_argument("--id_key", default="sampleId")
    args = parser.parse_args()

    data = load_data(args.input_path)
    model_class = resolve_summarizer_class(args.type)
    summarizer = model_class(args.model_path)

    out_data = []
    batch = []
    batch_size = args.batch_size
    progress_bar = tqdm(total=len(data), desc="Processing")
    for item in data:
        if len(batch) == batch_size:
            outputs = summarizer.greedy([_item[args.input_key] for _item in batch], block_n_grams=args.block_n_grams)
            for output, _item in zip(outputs, batch):
                out_data.append({
                    args.id_key: _item[args.id_key],
                    args.input_key: _item[args.input_key],
                    args.output_key: output
                })
                progress_bar.update(1)
            batch = []
        batch.append(item)
    
    if len(batch) > 0:
        outputs = summarizer.greedy(
            [_item[args.input_key] for _item in batch],
            block_n_grams=args.block_n_grams,
            max_length=args.max_length
        )
        for output, _item in zip(outputs, batch):
            out_data.append({
                args.id_key: _item[args.id_key],
                args.input_key: _item[args.input_key],
                args.output_key: output
            })
            progress_bar.update(1)
    
    with open(args.output_path, "w") as writer:
        for item in tqdm(out_data, desc="Writing"):
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
