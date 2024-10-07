import json
import argparse
from tqdm import tqdm
from evaluate import load

from libs.data_helpers.bytedataset import ByteDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_data_path", default="data/vietnews-hf/full/bytedataset/test_desegmented"
    )
    parser.add_argument("--data_path", default="annotated_data.jsonl")
    parser.add_argument("--output_file", default="bertscore_ranking.txt")
    parser.add_argument("--model", default="vinai/phobert-base")
    parser.add_argument("--num_layers", type=int, default=12)

    args = parser.parse_args()

    test_dataset = ByteDataset(data_path=args.test_data_path)
    ref_data = []
    iterator = iter(test_dataset)
    for _ in range(200):
        ref_data.append(next(iterator))

    data = []
    with open(args.data_path, "r", encoding="utf-8") as reader:
        for line in reader:
            data.append(json.loads(line.strip()))

    bertscore = load("bertscore")

    references = []
    hypotheses = []
    item_boundaries = []
    count = 0
    for i in range(len(data)):
        item_ref = ref_data[i]["abstract"]
        items = data[i]["outputs"]
        item_hypotheses = [item["content"] for item in items]
        item_refs = [item_ref] * len(items)
        item_boundaries.append((count, count + len(items)))
        count += len(items)
        references.extend(item_refs)
        hypotheses.extend(item_hypotheses)

    results = bertscore.compute(
        predictions=hypotheses,
        references=references,
        num_layers=args.num_layers,
        model_type=args.model,
    )

    f1_scores = results["f1"]

    with open(args.output_file, "w", encoding="utf-8") as writer:
        for start, end in item_boundaries:
            scores = f1_scores[start:end]
            print(scores)
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
