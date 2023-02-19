import os
import re
import json
import argparse
from tqdm import tqdm
from typing import Text, List, Any, Dict

import torch
from torch import nn
from torch.utils.data import Dataset, IterableDataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers.modeling_outputs import Seq2SeqModelOutput


class Reporter(object):
    pass


class EncoderReporter(Reporter):
    @staticmethod
    def message(model_name, isotropy, counter):
        msg = "Encoder isotropy of '{}' is {}\n".format(model_name, isotropy)
        msg += "Calculated on {} sentences.\n".format(counter)
        return msg


class DecoderReporter(Reporter):
    @staticmethod
    def message(model_name, isotropy, counter):
        msg = "Decoder isotropy of '{}' is {}\n".format(model_name, isotropy)
        msg += "Calculated on {} sentences.\n".format(counter)
        return msg


class EncoderDecoderReporter(Reporter):
    @staticmethod
    def message(model_name, isotropy, counter):
        msg = "Encoder decoder isotropy of '{}' is {}\n".format(model_name, isotropy)
        msg += "Calculated on {} seq-seq pairs.\n".format(counter)
        return msg


class CorpusDataset(IterableDataset):
    def __init__(self, data_path):
        self.data_path = data_path
    
    def __iter__(self):
        with open(self.data_path, "r") as reader:
            for line in reader:
                yield json.loads(line.strip())['sentence']


class Seq2SeqDataset(IterableDataset):
    def __init__(self, data_path):
        self.data_path = data_path
    
    def __iter__(self):
        with open(self.data_path, "r") as reader:
            for line in reader:
                yield json.loads(line.strip())


class AbmusuDataset(IterableDataset):
    def __init__(self, data_path):
        self.data_path = data_path
    
    def __iter__(self):
        with open(self.data_path, "r") as reader:
            for line in reader:
                item = json.loads(line.strip())
                multi_docs = [doc['raw_text'] for doc in item['single_documents']]
                concat_doc = " ".join(multi_docs)
                yield {'in': concat_doc, 'out': item['summary']}


class VietnameseMDSDataset(IterableDataset):
    def __init__(self, data_path):
        self.data_path = data_path
    
    def __iter__(self):
        with open(self.data_path, "r") as reader:
            for line in reader:
                item = json.loads(line.strip())
                multi_docs = [doc['raw_text'] for doc in item['single_documents']]
                concat_doc = " ".join(multi_docs)
                yield {'in': concat_doc, 'out': item['summary']}


class VimsDataset(IterableDataset):
    def __init__(self, data_path):
        self.data_path = data_path
    
    def __iter__(self):
        stack = [self.data_path]
        while stack:
            prefix = stack.pop()
            if os.path.isfile(prefix):
                with open(prefix) as reader:
                    yield self._get_item(reader.read())
            if not os.path.isdir(prefix):
                continue
            paths = os.listdir(prefix)
            paths = [os.path.join(prefix, p) for p in paths]
            subdirs = [p for p in paths if os.path.isdir(p)]
            subdirs = sorted(subdirs, reverse=True)
            files = [p for p in paths if os.path.isfile(p)]
            files = sorted(files, reverse=True)
            stack.extend(files)
            stack.extend(subdirs)

    def _get_item(self, content):
        match = re.search(r"Summary: (.*?)\nContent:\n(.*)", content, re.DOTALL)
        return {'in': match.group(2).strip(), 'out': match.group(1).strip()}


class VietnewsDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_size = len(os.listdir(self.data_path))
    
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        file_name = "{:06d}.txt.seg".format(idx + 1)
        with open(os.path.join(self.data_path, file_name), "r") as reader:
            content = reader.read()
        content = re.sub(r"[\r\t\f\v  ]+\n", "\n", content) # remove trailing whitespaces at each line
        parts = content.split("\n\n")
        summary = parts[1]
        document = parts[2]
        return {'in': document, 'out': summary}


class CNNDMDataset(IterableDataset):
    def __init__(self, data_path):
        self.data_path = data_path
    
    def __iter__(self):
        files = os.listdir(self.data_path)
        files = sorted(files)
        files = [os.path.join(self.data_path, f) for f in files]
        for f in files:
            data = torch.load(f)
            for item in data:
                src_txt = " ".join(item['src_txt'])
                tgt_txt = item['tgt_txt']
                yield {'in': src_txt, 'out': tgt_txt}


def shift_right(tensor: torch.Tensor, shift_id: int):
    shifted_tensor = tensor.new_zeros(tensor.shape)
    shifted_tensor[..., 1:] = tensor[..., :-1].clone()
    shifted_tensor[..., 0] = shift_id
    return shifted_tensor


def prepend(tensor: torch.Tensor, prepend_id: int):
    *ds, dn = tensor.size()
    shifted_tensor = tensor.new_zeros(*ds, dn + 1)
    shifted_tensor[..., 1:] = tensor.clone()
    shifted_tensor[..., 0] = prepend_id
    return shifted_tensor


def get_enc_collate_fn(tokenizer, max_len=1024):
    def collate_fn(items: List[Text]):
        batch = tokenizer(items, return_tensors='pt', padding=True)
        return batch
    return collate_fn


def get_dec_collate_fn(tokenizer, max_len=1024):
    def collate_fn(items: List[Text]):
        batch = tokenizer(items, return_tensors='pt',
            padding=True, truncation=True, max_length=max_len, add_special_tokens=False)
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        return {'input_ids': input_ids, 'attention_mask': attention_mask}
    return collate_fn


def get_seq2seq_collate_fn(tokenizer, model, max_len=1024):
    def collate_fn(items: List[Any]):
        documents = []
        summaries = []
        for item in items:
            documents.append(item['in'])
            summaries.append(item['out'])
        document_inputs = tokenizer(documents, max_length=max_len - 1, padding=True, truncation=True, return_tensors='pt')
        summary_inputs = tokenizer(summaries, padding=True, return_tensors='pt', add_special_tokens=False)
        decoder_start_token_id = model.config.decoder_start_token_id

        shifted_summary_input_ids = prepend(summary_inputs.input_ids, decoder_start_token_id)
        shifted_summary_attention_mask = prepend(summary_inputs.attention_mask, 1)
        return {
            "input_ids": document_inputs.input_ids,
            "attention_mask": document_inputs.attention_mask,
            "decoder_input_ids": shifted_summary_input_ids,
            "decoder_attention_mask": shifted_summary_attention_mask
        }
    return collate_fn


class IsotropyMeasurer(object):
    def __init__(self):
        self.self_similarity = 0.0
        self.counter = 0
    
    def reset_metric(self):
        self.self_similarity = 0.0
        self.counter = 0

    def calculate_self_similarity(
        self,
        model: nn.Module,
        batch: List[Any]
    ):
        batch = {k: v.to(device) for k, v in batch.items()}
        model.eval()
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True, return_dict=True)
            if isinstance(outputs, Seq2SeqModelOutput):
                sequence_output = outputs.decoder_hidden_states[-1] # [bsz, seq_len, hidden_size]
            else:
                sequence_output = outputs.last_hidden_state
        
        # calculate self similarity
        batch_size = sequence_output.size(0)
        if 'decoder_attention_mask' in batch:
            batch_attention_mask = batch['decoder_attention_mask']
        else:
            batch_attention_mask = batch['attention_mask']
        for i in range(batch_size):
            token_embeddings = sequence_output[i] # [seq_len, hidden_size]
            active_mask = batch_attention_mask[i].to(torch.bool) # [seq_len]
            active_token_embeddings = token_embeddings[active_mask] # [actual_seq_len, hidden_size]
            active_token_embeddings = torch.nn.functional.normalize(active_token_embeddings)
            actual_seq_length = active_token_embeddings.size(0)
            if actual_seq_length < 2:
                continue
            token_sim_matrix = torch.matmul(active_token_embeddings, active_token_embeddings.T) # [actual_seq_len, actual_seq_len]
            active_sim_matrix_mask = torch.triu(token_sim_matrix, diagonal=1).to(torch.bool)
            active_sim_matrix = torch.where(active_sim_matrix_mask, token_sim_matrix, 0.0)
            denominator = actual_seq_length * (actual_seq_length - 1) / 2
            sentence_self_similarity = active_sim_matrix.sum() / denominator
            self.self_similarity += sentence_self_similarity
            self.counter += 1

    @property
    def isotropy(self):
        return 1 - self.self_similarity / self.counter


GENERAL_MAP_TABLE = {
    "enc": {
        "get_collate_fn": get_enc_collate_fn,
        "dataset_class": CorpusDataset,
        "reporter": EncoderReporter
    },
    "dec": {
        "get_collate_fn": get_dec_collate_fn,
        "dataset_class": CorpusDataset,
        "reporter": DecoderReporter
    },
    "enc-dec": {
        "get_collate_fn": get_seq2seq_collate_fn,
        "dataset_class": "unresolved",
        "reporter": EncoderDecoderReporter
    }
}

DATASET_MAP_TABLE = {
    "abmusu": AbmusuDataset,
    "vimds": VietnameseMDSDataset,
    "vims": VimsDataset,
    "vietnews": VietnewsDataset,
    "cnndm": CNNDMDataset
}


def calculate_isotropy(model, tokenizer, args):
    assert args.model_type in ["enc", "dec", "enc-dec"]

    # get collate fn
    get_collate_fn = GENERAL_MAP_TABLE[args.model_type]["get_collate_fn"]
    if args.model_type in {"enc", "dec"}:
        if args.max_len:
            collate_fn = get_collate_fn(tokenizer, args.max_len)
        else:
            collate_fn = get_collate_fn(tokenizer)
    else:
        if args.max_len:
            collate_fn = get_collate_fn(tokenizer, model, args.max_len)
        else:
            collate_fn = get_collate_fn(tokenizer, model)

    # get reporter
    reporter = GENERAL_MAP_TABLE[args.model_type]["reporter"]

    # get dataset class
    dataset_class = GENERAL_MAP_TABLE[args.model_type]["dataset_class"]
    if dataset_class == "unresolved":
        dataset_class = DATASET_MAP_TABLE[args.dataset_name]

    # instantiate dataset and dataloader
    dataset = dataset_class(args.data_path)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False
    )

    # instantiate isotropy measurer
    measurer = IsotropyMeasurer()

    # loop to calculate isotropy
    for i, batch in tqdm(enumerate(dataloader)):
        measurer.calculate_self_similarity(model, batch)
        if (i + 1) % args.batch_report == 0:
            with open(args.output_path, "w") as writer:
                writer.write(reporter.message(args.model_path, measurer.isotropy, measurer.counter))
    
    with open(args.output_path, "w") as writer:
        writer.write(reporter.message(args.model_path, measurer.isotropy, measurer.counter))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--dataset-name", default="abmusu",
                        choices=["abmusu", "vimds", "vietnews", "vims", "cnndm"])
    parser.add_argument("--model-type", default="enc-dec",
                        choices=['enc-dec', 'enc', 'dec', 'separate-enc-dec'])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tokenizer-path", default="VietAI/vit5-base-vietnews-summarization")
    parser.add_argument("--model-path", default="VietAI/vit5-base-vietnews-summarization")
    parser.add_argument("--output-path", default="output.txt")
    parser.add_argument("--batch-report", default=10, type=int)
    parser.add_argument("--max-len", default=None, type=int)
    args = parser.parse_args()

    with open(args.output_path, "w"): # test errors on opening file to write
        pass

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model = AutoModel.from_pretrained(args.model_path)
    global device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    if args.model_type == "separate-enc-dec":
        dataset = CorpusDataset(args.data_path)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=get_enc_collate_fn(tokenizer),
            drop_last=False
        )
        enc_measurer = IsotropyMeasurer()
        dec_measurer = IsotropyMeasurer()
        enc = model.encoder
        dec = model.decoder

        for i, batch in tqdm(enumerate(dataloader)):
            enc_measurer.calculate_self_similarity(enc, batch)
            dec_batch = {**batch, 'input_ids': shift_right(batch['input_ids'], dec.config.decoder_start_token_id)}
            dec_measurer.calculate_self_similarity(dec, dec_batch)
            if (i + 1) % args.batch_report:
                with open(args.output_path, "w") as writer:
                    writer.write("Encoder isotropy of '{}' = {}\n".format(args.model_path, enc_measurer.isotropy))
                    writer.write("Decoder isotropy of '{}' = {}\n".format(args.model_path, dec_measurer.isotropy))
                    assert enc_measurer.counter == dec_measurer.counter
                    writer.write("Calculated on #{} sentences.".format(enc_measurer.counter))
    else:
        calculate_isotropy(model, tokenizer, args)


if __name__ == "__main__":
    main()
