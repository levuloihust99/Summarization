import json
import argparse
from tqdm import tqdm

from rlhf.nn.reward import T5CrossEncoderReward
from rlhf.nn.device import init_device

init_device(None)


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
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--data_path", default="data.jsonl")
    parser.add_argument("--output_file", default="agreement.txt")
    parser.add_argument("--ref_ranking", nargs="+", default=["ranking.txt"])
    parser.add_argument("--use_manual_label", type=eval, default=False)
    parser.add_argument("--hyp_ranking", default=None)
    parser.add_argument(
        "--pretrained_path", default="VietAI/vit5-base-vietnews-summarization"
    )
    parser.add_argument("--sep_token", default="<extra_id_0>")
    args = parser.parse_args()

    if args.hyp_ranking is None:
        reward_model = T5CrossEncoderReward(
            ckpt_path=args.ckpt_path,
            pretrained_path=args.pretrained_path,
            sep_token=args.sep_token,
        )

        data = []
        with open(args.data_path, "r", encoding="utf-8") as reader:
            for line in reader:
                data.append(json.loads(line.strip()))
        L = len(data)
    else:
        hyp_ranking = read_ranking(args.hyp_ranking)
        L = len(hyp_ranking)

    ref_rankings = []
    if not args.use_manual_label:
        for ranking_file in args.ref_ranking:
            ref_rankings.append(read_ranking(ranking_file))
    else:
        ref_ranking = []
        for i in range(L):
            if args.hyp_ranking is None:
                ref_ranking.append(list(range(1, len(data[i]) + 1)))
            else:
                ref_ranking.append(list(range(1, len(hyp_ranking[i]) + 1)))
        ref_rankings.append(ref_ranking)

    with open(args.output_file, "w", encoding="utf-8") as writer:
        progress_bar = tqdm(total=L)
        for idx in range(L):
            if args.hyp_ranking is None:
                item = data[idx]
                doc = item["input"]
                scores = []
                for out in item["outputs"]:
                    hyp = out["content"]
                    score = reward_model.cal_reward(
                        doc=doc, hyp=hyp, ref=None, training=False
                    )
                    scores.append(score)
                scores_with_idxs = []
                for i, score in enumerate(scores):
                    scores_with_idxs.append((i + 1, score))
                sorted_scores_with_idxs = sorted(
                    scores_with_idxs, key=lambda x: x[1], reverse=True
                )
                ranking = [x[0] for x in sorted_scores_with_idxs]
            else:
                ranking = hyp_ranking[idx]
            agreement = calculate_agreement(
                ranking, [ref_ranking[idx] for ref_ranking in ref_rankings]
            )
            writer.write("{}\n".format(agreement))
            progress_bar.update(1)


if __name__ == "__main__":
    main()
