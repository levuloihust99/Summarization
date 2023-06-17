import re
import argparse
import string
import torch
from tqdm import tqdm
from nltk import sent_tokenize, word_tokenize
from typing import Text, Dict, List
from compare_mt.rouge.rouge_scorer import RougeScorer

from torch.utils.data import DataLoader

from rlhf.mapping import resolve_tokenizer_class, resolve_model_class
from rlhf.infer import get_data_collator
from libs.utils.rouge_calculator import _rouge_n_sentence_level, _f_score
from libs.data_helpers.bytedataset import ByteDataset

punc_patt = re.compile(f"[{re.escape(string.punctuation)}]")


def cal_rouge_manually(hyp: Text, ref: Text, ns: List[int] = [1]):
    hyp = punc_patt.sub(" ", hyp)
    ref = punc_patt.sub(" ", ref)
    hyp_tokens = hyp.lower().split()
    ref_tokens = ref.lower().split()
    
    score = {}
    for n in ns:
        score[n] = _rouge_n_sentence_level(hyp_tokens, ref_tokens, n).to_score(alpha=0.5)
    
    flatten_score = {}
    for n in ns:
        flatten_score[f"rouge{n}-p"] = score[n]["p"]
        flatten_score[f"rouge{n}-r"] = score[n]["r"]
        flatten_score[f"rouge{n}-f1"] = score[n]["f"]

    return flatten_score


def evaluate_generator(
    model,
    dataloader,
    tokenizer,
    device,
    generation_kwargs,
    *,
    manual: bool = True,
    input_name: Text = "document",
    output_name: Text = "summary"
):
    if manual is True:
        return evaluate_generator_manually(
            model,
            dataloader,
            tokenizer,
            device,
            input_name,
            output_name,
            generation_kwargs
        )
    else:
        return evaluate_generator_brio(
            model,
            dataloader,
            tokenizer,
            device,
            input_name,
            output_name,
            generation_kwargs
        )


def evaluate_generator_manually(
    model,
    dataloader,
    tokenizer,
    device,
    input_name,
    output_name,
    generation_kwargs
):
    model.eval()
    hypotheses = []
    references = []
    scores = []
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
            for hyp, ref in zip(batch_hyps, batch_refs):
                score = cal_rouge_manually(hyp, ref, [1, 2])
                scores.append(score)

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

    return {
        "eval/rouge1-precision": avg_rouge1_p,
        "eval/rouge1-recall": avg_rouge1_r,
        "eval/rouge1-f1": avg_rouge1_f1,
        "eval/rouge2-precision": avg_rouge2_p,
        "eval/rouge2-recall": avg_rouge2_r,
        "eval/rouge2-f1": avg_rouge2_f1
    }


def process(x):
    return sent_tokenize(" ".join(word_tokenize(x.strip())))


def evaluate_generator_brio(
    model,
    dataloader,
    tokenizer,
    device,
    input_name,
    output_name,
    generation_kwargs
):
    rouge_scorer = RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    hypotheses = []
    references = []
    scores = []
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):
            input_ids = batch["input_ids"]
            input_ids = torch.stack(input_ids, dim=0).to(device)
            output = model.generate(input_ids=input_ids, **generation_kwargs)
            batch_hyps = [
                tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                for output_ids in output
            ]
            batch_refs = batch[output_name]
            for hyp, ref in zip(batch_hyps, batch_refs):
                hyp = hyp.replace("\n", " ")
                hyp = process(hyp)
                ref = process(ref)
                score = rouge_scorer.score("\n".join(ref), "\n".join(hyp))
                scores.append(score)

            hypotheses.extend(batch_hyps)
            references.extend(batch_refs)

    rouge1_p = []
    rouge1_r = []
    rouge1_f = []
    rouge2_p = []
    rouge2_r = []
    rouge2_f = []
    for score in scores:
        rouge1_p.append(score["rouge1"].precision)
        rouge1_r.append(score["rouge1"].recall)
        rouge1_f.append(score["rouge1"].fmeasure)
        rouge2_p.append(score["rouge2"].precision)
        rouge2_r.append(score["rouge2"].recall)
        rouge2_f.append(score["rouge2"].fmeasure)
    count = len(scores)

    avg_rouge1_p = sum(rouge1_p) / count
    avg_rouge1_r = sum(rouge1_r) / count
    avg_rouge1_f = _f_score(avg_rouge1_p, avg_rouge1_r, alpha=0.5)
    avg_rouge2_p = sum(rouge2_p) / count
    avg_rouge2_r = sum(rouge2_r) / count
    avg_rouge2_f = _f_score(avg_rouge2_p, avg_rouge2_r, alpha=0.5)

    return {
        "eval/rouge1-precision": avg_rouge1_p,
        "eval/rouge1-recall": avg_rouge1_r,
        "eval/rouge1-f1": avg_rouge1_f,
        "eval/rouge2-precision": avg_rouge2_p,
        "eval/rouge2-recall": avg_rouge2_r,
        "eval/rouge2-f1": avg_rouge2_f
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--tokenizer_path", default="VietAI/vit5-base-vietnews-summarization")
    parser.add_argument("--tokenizer_class", default="AutoTokenizer")
    parser.add_argument("--model_path", default="VietAI/vit5-base-vietnews-summarization")
    parser.add_argument("--model_class", default="AutoModelForSeq2SeqLM")
    parser.add_argument("--max_output_len", type=int, default=200)
    parser.add_argument("--input_name", default="document")
    parser.add_argument("--output_name", default="summary")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_stat_file", default="stats.txt")
    parser.add_argument("--do_sample", action="store_true", default=False)
    args = parser.parse_args()

    tokenizer_class = resolve_tokenizer_class(args.tokenizer_class)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_class = resolve_model_class(args.model_class)
    model = model_class.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    default_gen_kwargs = {
        "no_repeat_ngram_size": 3,
        "max_new_tokens": args.max_output_len,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": args.do_sample,
    }

    dataset = ByteDataset(args.data_path, 6)
    data_collator = get_data_collator(
        tokenizer,
        input_name=args.input_name,
        output_name=args.output_name,
        max_input_len=None,
        max_output_len=None
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator
    )

    stats = evaluate_generator(
        model=model,
        dataloader=dataloader,
        tokenizer=tokenizer,
        device=device,
        generation_kwargs=default_gen_kwargs,
        manual=True,
        input_name=args.input_name,
        output_name=args.output_name
    )
    metrics = [m[len("eval/"):] for m in stats]
    max_metric_name_len = max(len(m) for m in metrics)
    output = ""
    for m, v in zip(metrics, stats.values()):
        row = (m + " " * (max_metric_name_len - len(m) + 1) + "= ")
        row = "{}{}".format(row, v)
        output += row + "\n"

    print(output)
    with open(args.output_stat_file, "w") as writer:
        writer.write(output)


if __name__ == "__main__":
    main()
