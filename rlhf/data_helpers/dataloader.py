import torch
from typing import List, Dict, Optional


def get_data_collator(
    tokenizer,
    input_name,
    output_name,
    max_input_len: Optional[int] = None,
    max_output_len: Optional[int] = None,
):
    def collate_fn(items: List[Dict[str, str]]):
        if len(items) == 0:
            return {
                "ids": [],
                "input_ids": [],
                "decoder_input_ids": [],
                input_name: [],
                output_name: []
            }

        documents = []
        summaries = []
        ids = []
        for item in items:
            documents.append(item[input_name])
            summaries.append(item[output_name])
            ids.append(item.get("id", None) or item.get("guid", None))

        pretokenized = items[0].get("pretokenized")
        if pretokenized:
            encoded_documents = []
            encoded_summaries = []
            for item in items:
                doc_input_ids = item["pretokenized"][input_name]
                summ_input_ids = item["pretokenized"][output_name]
                if max_input_len and max_input_len > 0:
                    doc_input_ids = doc_input_ids[:max_input_len]
                    doc_input_ids[-1] = tokenizer.eos_token_id
                if max_output_len and max_output_len > 0:
                    summ_input_ids = summ_input_ids[:max_output_len]
                    summ_input_ids[-1] = tokenizer.eos_token_id
                encoded_documents.append(doc_input_ids)
                encoded_summaries.append(summ_input_ids)
            return {
                "ids": ids,
                "input_ids": [torch.tensor(v) for v in encoded_documents],
                "decoder_input_ids": [torch.tensor(v) for v in encoded_summaries],
                input_name: documents,
                output_name: summaries
            }

        encoded_documents = tokenizer(
            documents,
            padding=False,
            truncation=True,
            max_length=max_input_len,
        )
        encoded_summaries = tokenizer(
            summaries,
            padding=False,
            truncation=True,
            max_length=max_output_len
        )

        return {
            "ids": ids,
            "input_ids": [torch.tensor(v) for v in encoded_documents.input_ids],
            "decoder_input_ids": [torch.tensor(v) for v in encoded_summaries.input_ids],
            input_name: documents,
            output_name: summaries
        }
    return collate_fn
