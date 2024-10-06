import json
import argparse
from tqdm import tqdm

from rlhf.nn.reward import T5CrossEncoderReward
from rlhf.nn.device import init_device

init_device(None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--data_path", default="data.jsonl")
    parser.add_argument("--output_file", default="hyp_ranking.txt")
    parser.add_argument(
        "--pretrained_path", default="VietAI/vit5-base-vietnews-summarization"
    )
    parser.add_argument("--sep_token", default="<extra_id_0>")
    args = parser.parse_args()

    reward_model = T5CrossEncoderReward(
        ckpt_path=args.ckpt_path,
        pretrained_path=args.pretrained_path,
        sep_token=args.sep_token,
    )

    data = []
    with open(args.data_path, "r", encoding="utf-8") as reader:
        for line in reader:
            data.append(json.loads(line.strip()))

    with open(args.output_file, "w", encoding="utf-8") as writer:
        for idx, item in enumerate(tqdm(data)):
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
            writer.write("{}\n".format(", ".join([str(_i) for _i in ranking])))
            writer.flush()


if __name__ == "__main__":
    main()
