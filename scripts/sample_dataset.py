import os
import json
import random
import pickle
import argparse

from tqdm import tqdm

from libs.data_helpers.bytedataset import ByteDataset
from libs.data_helpers.utils import create_bytedataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--input_path", "-i", required=True,
                        help="Path to the input bytedataset.")
    parser.add_argument("--output_path", "-o", required=True,
                        help="Path to the output dataset.")
    parser.add_argument("--output_format",
                        choices=["jsonlines", "bytedataset", "both"],
                        default="jsonlines")
    parser.add_argument("--sample_size", type=int, default=1000)
    args = parser.parse_args()

    random.seed(args.seed)

    input_dataset = ByteDataset(data_path=args.input_path, idx_record_size=6)
    dataset_size = len(input_dataset)

    idxs = list(range(dataset_size))
    random.shuffle(idxs)
    sampled_idxs = idxs[:args.sample_size]

    out_data = []
    for idx in sampled_idxs:
        out_data.append(input_dataset[idx])

    output_jsonlines = True if args.output_format == "jsonlines" else False
    output_bytedataset = True if args.output_format == "bytedataset" else False
    if output_jsonlines:
        output_dir = os.path.dirname(args.output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(args.output_path, "w") as writer:
            for item in out_data:
                writer.write(json.dumps(item, ensure_ascii=False) + "\n")
    if output_bytedataset:
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        create_bytedataset(
            output_dir=args.output_path,
            data=out_data
        )


if __name__ == "__main__":
    main()
