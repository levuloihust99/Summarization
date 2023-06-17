import re
import string
import torch
from typing import Text
from transformers import AutoModel, AutoTokenizer

from libs.utils.rouge_calculator import _rouge_n_sentence_level

punc_patt = re.compile(f"[{re.escape(string.punctuation)}]")


def remove_punctuation(text: Text) -> Text:
    text = punc_patt.sub(" ", text)
    return text


class Rouge1F1Reward(object):
    def __init__(self):
        self.total_reward = 0.0
        self.n_samples = 0

    def cal_reward(self, hyp: Text, ref: Text):
        hyp = remove_punctuation(hyp)
        ref = remove_punctuation(ref)
        hyp_tokens = hyp.lower().split()
        ref_tokens = ref.lower().split()
        score = _rouge_n_sentence_level(hyp_tokens, ref_tokens, 1).to_score(alpha=0.5)
        reward = score["f"]
        self.n_samples += 1
        self.total_reward += reward
    
    def get_avg_reward(self):
        if self.n_samples == 0:
            return 0.0
        return self.total_reward / self.n_samples


class SentenceEmbeddingSimilarityReward(object):
    def __init__(self, sim_model):
        self.tokenizer = AutoTokenizer.from_pretrained(sim_model)
        self.sim_model = AutoModel.from_pretrained(sim_model)
        self.sim_model.eval()
        self.total_reward = 0.0
        self.n_samples = 0
    
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def cal_reward(self, hyp: Text, ref: Text):
        inputs = self.tokenizer([hyp, ref], padding=True, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        with torch.no_grad():
            outputs = self.sim_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        embeddings = self.mean_pooling(outputs, attention_mask)
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        reward = torch.sum(embeddings[0] * embeddings[1]).item() * 5
        self.n_samples += 1
        self.total_reward += reward
        return reward

    def get_avg_reward(self):
        if self.n_samples == 0:
            return 0.0
        return self.total_reward / self.n_samples
