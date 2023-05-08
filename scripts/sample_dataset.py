import os
import json
import random
import pickle
import argparse

from tqdm import tqdm

from libs.data_helpers.bytedataset import ByteDataset


def write_bytedataset(
    output_path,
    output_data
):
    idx_writer = open(os.path.join(output_path, "idxs.pkl"), "wb")
    dataset_size_place_holder = (0).to_bytes(4, 'big', signed=False)
    idx_writer.write(dataset_size_place_holder)

    data_writer = open(os.path.join(output_path, "data.pkl"), "wb")

    num_records = 0
    for item in tqdm(output_data):
        idx_writer.write(data_writer.tell().to_bytes(6, 'big', signed=False))
        pickle.dump(item, data_writer)
        num_records += 1
        
    data_writer.close()
        
    idx_writer.seek(0, 0)
    idx_writer.write(num_records.to_bytes(4, 'big', signed=False))
    idx_writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--input-path", "-i", required=True,
                        help="Path to the input bytedataset.")
    parser.add_argument("--output-path", "-o", required=True,
                        help="Path to the output dataset.")
    parser.add_argument("--output-format",
                        choices=["jsonlines", "bytedataset", "both"],
                        default="jsonlines")
    parser.add_argument("--sample-size", type=int, default=1000)
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
        write_bytedataset(
            output_path=args.output_path,
            output_data=out_data
        )


if __name__ == "__main__":
    main()
