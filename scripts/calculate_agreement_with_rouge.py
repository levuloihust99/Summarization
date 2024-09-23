import json
import argparse
from tqdm import tqdm

from libs.utils.rouge_calculator import rouge_n_sentence_level
from libs.data_helpers.bytedataset import ByteDataset


def calculate_agreement(ranking: list[int], ref_rankings: list[list[int]]):
    total = 0
    matches = 0
    ref_ranking_maps = [
        {v: k for k, v in enumerate(ref_ranking)} for ref_ranking in ref_rankings
    ]
    for i in range(len(ranking) - 1):
        for j in range(i + 1, len(ranking)):
            rank_i = ranking[i]
            rank_j = ranking[j]

            ref_compare = None
            all_agree = True
            for k, ref_ranking_map in enumerate(ref_ranking_maps):
                if k == 0:
                    ref_compare = ref_ranking_map[rank_i] - ref_ranking_map[rank_j]
                else:
                    current_compare = ref_ranking_map[rank_i] - ref_ranking_map[rank_j]
                    if current_compare * ref_compare < 0:
                        all_agree = False
                        break
            if all_agree is False:
                continue

            total += 1
            if ref_compare < 0:
                matches += 1

    return matches / total


def read_ranking(ranking_file):
    rankings = []
    with open(ranking_file, "r", encoding="utf-8") as reader:
        for line in reader:
            orders = line.strip().split(",")
            orders = [int(i.strip()) for i in orders]
            rankings.append(orders)
    return rankings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", default="data/vietnews-hf/full/bytedataset/test_desegmented")
    parser.add_argument("--data_path", default="data.jsonl")
    parser.add_argument("--output_file", default="agreement.txt")
    parser.add_argument("--ref_ranking", nargs="+", default=["ranking.txt"])
    args = parser.parse_args()

    test_dataset = ByteDataset(data_path=args.test_data_path)
    ref_data = []
    iterator = iter(test_dataset)
    for _ in range(200):
        ref_data.append(next(iterator))

    ref_rankings = []
    for ranking_file in args.ref_ranking:
        ref_rankings.append(read_ranking(ranking_file))

    data = []
    with open(args.data_path, "r", encoding="utf-8") as reader:
        for line in reader:
            data.append(json.loads(line.strip()))

    with open(args.output_file, "w", encoding="utf-8") as writer:
        for idx, item in enumerate(tqdm(data)):
            ref_output = ref_data[idx]["abstract"]
            scores = []
            for out in item["outputs"]:
                hyp = out["content"]
                score = rouge_n_sentence_level(hyp=hyp, ref=ref_output, ns=[2])
                scores.append(score["rouge2"]["f"])
            scores_with_idxs = []
            for i, score in enumerate(scores):
                scores_with_idxs.append((i + 1, score))
            sorted_scores_with_idxs = sorted(
                scores_with_idxs, key=lambda x: x[1], reverse=True
            )
            ranking = [x[0] for x in sorted_scores_with_idxs]
            agreement = calculate_agreement(
                ranking, [ref_ranking[idx] for ref_ranking in ref_rankings]
            )
            writer.write("{}\n".format(agreement))


if __name__ == "__main__":
    main()
