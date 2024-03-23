import os
import pickle

from tqdm import tqdm
from typing import List, Any

from libs.data_helpers.bytedataset import ByteDataset
from libs.data_helpers.utils import create_bytedataset


def pretokenize(bytedataset_path, tokenizer, input_name: str, output_name: str):
    pretokenize_path = os.path.join(bytedataset_path, "pretokenized")
    if os.path.exists(pretokenize_path):
        return

    dataset = ByteDataset(bytedataset_path)
    data = []
    for item in tqdm(dataset, desc="Pretokenizing"):
        document = item[input_name]
        document_token_ids = tokenizer(document).input_ids
        summary = item[output_name]
        summary_token_ids = tokenizer(summary).input_ids
        pretokenized = {
            input_name: document_token_ids,
            output_name: summary_token_ids
        }
        data.append({**item, "pretokenized": pretokenized})

    create_bytedataset(pretokenize_path, data)
