import os
import json
import logging
import argparse

from libs.utils.logging import do_setup_logging
from libs.data_helpers.bytedataset import ByteDataset

do_setup_logging(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idxs_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    dataset = ByteDataset(data_path=args.data_path)
    with open(args.idxs_path, "r", encoding="utf-8") as reader:
        idxs = json.load(reader)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    with open(os.path.join(args.output_path, "idxs.pkl"), "wb") as writer:
        writer.write((0).to_bytes(4, signed=False))
    with open(os.path.join(args.output_path, "data.pkl"), "wb") as writer:
        pass

    extracted_dataset = ByteDataset(args.output_path)

    for idx in idxs:
        extracted_dataset.add_item(dataset[idx])


if __name__ == "__main__":
    main()
