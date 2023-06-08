import torch
from typing import Text, Dict, List, Any, Callable, Optional

from torch.utils.data import DataLoader

from libs.data_helpers.bytedataset import ByteDataset


def get_labels(decoder_input_ids):
    out_decoder_input_ids = decoder_input_ids[:-1]
    labels = decoder_input_ids[1:]
    return out_decoder_input_ids, labels


def get_collate_fn(
    tokenizer,
    normalizer,
    decoder_start_token_id,
    input_transform: Optional[Callable] = None,
    output_transform: Optional[Callable] = None,
    max_input_len: int = None,
    max_output_len: int = None,
    input_name: Text = "prompt",
    output_name: Text = "completion"
):
    def collate_fn(items: List[Dict[Text, Any]]):
        if input_transform:
            items = [
                {
                    k: v if k != input_name
                    else input_transform(v)
                    for k, v in ex.items()
                } for ex in items
            ]
        if output_transform:
            items = [
                {
                    k: v if k != output_name
                    else output_transform(v)
                    for k, v in ex.items()
                } for ex in items
            ]
        batch_input_ids = []
        batch_output_ids = []
        batch_labels = []
        batch_max_input_len = 0
        batch_max_output_len = 0

        nonlocal max_input_len, max_output_len
        if max_input_len is None:
            max_input_len = float("inf")
        if max_output_len is None:
            max_output_len = float("inf")

        for item in items:
            input_ids = tokenizer.encode(normalizer(item[input_name]))
            output_ids = tokenizer.encode(normalizer(item[output_name]))

            if len(input_ids) > max_input_len:
                input_ids = input_ids[:max_input_len]
                input_ids[-1] = tokenizer.eos_token_id
            if len(output_ids) > max_output_len:
                output_ids = output_ids[:max_output_len]

            labels = output_ids
            output_ids = [decoder_start_token_id] + output_ids[:-1]

            if batch_max_input_len < len(input_ids):
                batch_max_input_len = len(input_ids)
            if batch_max_output_len < len(output_ids):
                batch_max_output_len = len(output_ids)

            batch_input_ids.append(input_ids)
            batch_output_ids.append(output_ids)
            batch_labels.append(labels)
        
        # padding
        padded_batch_input_ids = []
        padded_batch_output_ids = []
        padded_batch_labels = []
        batch_input_attn_mask = []
        batch_output_attn_mask = []

        for i in range(len(items)):
            # < prompt
            input_ids = batch_input_ids[i]
            input_pad_len = batch_max_input_len - len(input_ids)

            padded_input_ids = input_ids + [tokenizer.pad_token_id] * input_pad_len
            input_attn_mask = [1] * len(input_ids) + [0] * input_pad_len

            padded_batch_input_ids.append(padded_input_ids)
            batch_input_attn_mask.append(input_attn_mask)
            # prompt />

            # < completion
            output_ids = batch_output_ids[i]
            labels = batch_labels[i]
            output_pad_len = batch_max_output_len - len(output_ids)

            padded_output_ids = output_ids + [tokenizer.pad_token_id] * output_pad_len
            output_attn_mask = [1] * len(output_ids) + [0] * output_pad_len
            padded_labels = labels + [-100] * output_pad_len

            padded_batch_output_ids.append(padded_output_ids)
            padded_batch_labels.append(padded_labels)
            batch_output_attn_mask.append(output_attn_mask)
            # completion />

        batch = {
            "input_ids": torch.tensor(padded_batch_input_ids, dtype=torch.int64),
            "attention_mask": torch.tensor(batch_input_attn_mask, dtype=torch.int64),
            "decoder_input_ids": torch.tensor(padded_batch_output_ids, dtype=torch.int64),
            "decoder_attention_mask": torch.tensor(batch_output_attn_mask, dtype=torch.int64),
            "labels": torch.tensor(padded_batch_labels, dtype=torch.int64)
        }

        return batch

    return collate_fn
