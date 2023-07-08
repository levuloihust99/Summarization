import csv
import json
import torch
import argparse

from tqdm import tqdm
from typing import Text, Dict, Any, List, Optional

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from libs.data_helpers.bytedataset import ByteDataset


class JsonDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


def load_data(args):
    if args.data_format == "json":
        with open(args.data_path, "r") as reader:
            data = json.load(reader)
        dataset = JsonDataset(data)
    elif args.data_format == "jsonlines":
        data = []
        with open(args.data_path, "r") as reader:
            for line in reader:
                data.append(json.loads(line.strip()))
        dataset = JsonDataset(data)
    elif args.data_format == "bytedataset":
        dataset = ByteDataset(data_path=args.data_path, idx_record_size=6)
    else:
        raise Exception("Data format '{}' is not supported.".format(args.data_format))
    return dataset


def get_data_collator(
    tokenizer,
    input_name,
    output_name,
    max_input_len: Optional[int] = None,
    max_output_len: Optional[int] = None
):
    def collate_fn(items: List[Dict[Text, Any]]):
        input_texts = []
        output_texts = []
        for item in items:
            input_text = item[input_name]
            input_texts.append(input_text)
            output_text = item[output_name]
            output_texts.append(output_text)
        input_features = tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=max_input_len,
            return_tensors="pt"
        )
        output_features = tokenizer(
            output_texts,
            padding=True,
            truncation=True,
            max_length=max_output_len,
            return_tensors="pt"
        )
        batch = {
            "input_ids": input_features.input_ids,
            "attention_mask": input_features.attention_mask,
            "decoder_input_ids": output_features.input_ids,
            "decoder_attention_mask": output_features.attention_mask,
            input_name: input_texts,
            output_name: output_texts
        }
        return batch
    return collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_save_path", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--data_format", choices=["jsonlines", "bytedataset", "json"], default="jsonlines")
    parser.add_argument("--input_name", default="in")
    parser.add_argument("--output_name", default="out")
    parser.add_argument("--max_input_len", type=int, default=None)
    parser.add_argument("--max_output_len", type=int, default=None)
    parser.add_argument("--output_path", default="output.csv")
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_save_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_save_path)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()

    dataset = load_data(args)
    data_collator = get_data_collator(
        tokenizer,
        input_name=args.input_name,
        output_name=args.output_name
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator
    )

    generation_kwargs = {
        "no_repeat_ngram_size": 3,
        "max_new_tokens": args.max_output_len,
        "pad_token_id": tokenizer.eos_token_id
    }

    all_inputs = []
    all_hyp_texts = []
    all_ref_texts = []
    for batch in tqdm(data_loader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model.generate(
            input_ids=batch["input_ids"], **generation_kwargs)
        output_texts = [
            tokenizer.decode(
                output_ids, clean_up_tokenization_spaces=False, skip_special_tokens=True)
            for output_ids in outputs
        ]
        all_hyp_texts.extend(output_texts)
        all_ref_texts.extend(batch[args.output_name])
        all_inputs.extend(batch[args.input_name])
    
    with open(args.output_path, "w") as csvfile:
        csv_writer = csv.DictWriter(
            csvfile, fieldnames=["Input", "Reference", "Hypothesis"])
        csv_writer.writeheader()
        for inp, hyp, ref in zip(all_inputs, all_hyp_texts, all_ref_texts):
            csv_writer.writerow({"Input": inp, "Reference": ref, "Hypothesis": hyp})


if __name__ == "__main__":
    main()
