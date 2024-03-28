import torch
from tqdm import tqdm
from typing import Text, List, Optional, Literal

from rlhf.nn.device import device_manager
from rlhf.nn.reward.base import RewardModel
from libs.utils.rouge_calculator import rouge_n_sentence_level, _f_score


def cal_rouge(hyp: Text, ref: Text, ns: List[int] = [1]):
    scores = rouge_n_sentence_level(hyp, ref, ns)
    
    flatten_score = {}
    for n in ns:
        flatten_score[f"rouge{n}-p"] = scores[f"rouge{n}"]["p"]
        flatten_score[f"rouge{n}-r"] = scores[f"rouge{n}"]["r"]
        flatten_score[f"rouge{n}-f1"] = scores[f"rouge{n}"]["f"]

    return flatten_score


def evaluate_generator(
    model,
    dataloader,
    tokenizer,
    input_name,
    output_name,
    generation_kwargs,
    eval_mode: Literal["metric", "reward"] = "metric",
    reward_model: Optional[RewardModel] = None,
):
    model.eval()
    hypotheses = []
    references = []
    documents = []
    scores = []
    device = device_manager.device
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):
            input_ids = batch["input_ids"]
            if isinstance(batch["input_ids"], list):
                input_ids = torch.stack(input_ids, dim=0).to(device)
            else:
                input_ids = input_ids.to(device)
            if "attention_mask" in batch:
                attention_mask = batch["attention_mask"]
                if isinstance(attention_mask, list):
                    attention_mask = torch.stack(attention_mask, dim=0).to(device)
                else:
                    attention_mask = attention_mask.to(device)
            else:
                attention_mask = None
            output = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs)
            batch_hyps = [
                tokenizer.decode(output_ids, clean_up_tokenization_spaces=False, skip_special_tokens=True)
                for output_ids in output
            ]
            hypotheses.extend(batch_hyps)
            batch_refs = batch[output_name]
            references.extend(batch_refs)
            documents.extend(batch[input_name])

    model.train()
    for hyp, ref in zip(hypotheses, references):
        score = cal_rouge(hyp, ref, [1, 2])
        scores.append(score)

    if eval_mode in {"metric", "all"}:
        rouge1_precisions = []
        rouge1_recalls = []
        rouge2_precisions = []
        rouge2_recalls = []
        for score in scores:
            rouge1_precisions.append(score["rouge1-p"])
            rouge1_recalls.append(score["rouge1-r"])
            rouge2_precisions.append(score["rouge2-p"])
            rouge2_recalls.append(score["rouge2-r"])
        
        L = len(scores)
        avg_rouge1_p = sum(rouge1_precisions) / L
        avg_rouge1_r = sum(rouge1_recalls) / L
        avg_rouge1_f1 = _f_score(avg_rouge1_p, avg_rouge1_r, alpha=0.5)
        avg_rouge2_p = sum(rouge2_precisions) / L
        avg_rouge2_r = sum(rouge2_recalls) / L
        avg_rouge2_f1 = _f_score(avg_rouge2_p, avg_rouge2_r, alpha=0.5)

        metrics = {
            "eval/rouge1-precision": avg_rouge1_p,
            "eval/rouge1-recall": avg_rouge1_r,
            "eval/rouge1-f1": avg_rouge1_f1,
            "eval/rouge2-precision": avg_rouge2_p,
            "eval/rouge2-recall": avg_rouge2_r,
            "eval/rouge2-f1": avg_rouge2_f1
        }
        if eval_mode == "metric":
            return metrics

    rewards = []
    for doc, hyp, ref in zip(documents, hypotheses, references):
        r = reward_model.cal_reward(doc=doc, hyp=hyp, ref=ref, training=False)
        rewards.append(r)
    eval_reward = {
        "eval/reward": sum(rewards) / len(rewards)
    }

    if eval_mode == "reward":
        return eval_reward
    return {**metrics, **eval_reward}
