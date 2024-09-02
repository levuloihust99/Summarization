import json
import os
import random
import argparse
from libs.data_helpers.bytedataset import ByteDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to the bytedataset")
    parser.add_argument("--split_ratio", type=eval, default=[0.9, 0.1])
    parser.add_argument("--eval_num", type=int, default=1000)
    parser.add_argument("--train_idxs_path", required=True)
    parser.add_argument("--eval_idxs_path", required=True)
    args = parser.parse_args()

    dataset = ByteDataset(args.data_path)
    L = len(dataset)
    if L == 0:
        raise Exception("Dataset of length zero")

    idxs = list(range(L))
    random.shuffle(idxs)

    if args.eval_num > 0:
        train_num = L - args.eval_num
    else:
        train_num = max(int(args.split_ratio[0] * L), 1)

    train_idxs = idxs[:train_num]
    eval_idxs = idxs[train_num:]

    base_train_idxs_dir = os.path.abspath(os.path.dirname(args.train_idxs_path))
    if not os.path.exists(base_train_idxs_dir):
        os.makedirs(base_train_idxs_dir)
    with open(args.train_idxs_path, "w", encoding="utf-8") as writer:
        json.dump(train_idxs, writer)

    base_eval_idxs_dir = os.path.abspath(os.path.dirname(args.eval_idxs_path))
    if not os.path.exists(base_eval_idxs_dir):
        os.makedirs(base_eval_idxs_dir)
    with open(args.eval_idxs_path, "w", encoding="utf-8") as writer:
        json.dump(eval_idxs, writer)


if __name__ == "__main__":
    main()
