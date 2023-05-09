import torch
from tqdm import tqdm
from nltk import sent_tokenize, word_tokenize
from compare_mt.rouge.rouge_scorer import RougeScorer

from libs.utils.rouge_calculator import _f_score


def process(x):
    return sent_tokenize(" ".join(word_tokenize(x.strip())))

def evaluate_generator(
    model,
    dataloader,
    tokenizer,
    device,
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
            batch_refs = batch["summary"]
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
