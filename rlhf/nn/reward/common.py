import re
import torch
import string
import logging

from tqdm import tqdm
from typing import Text, Dict, Literal
from transformers import AutoModel, AutoTokenizer

from .base import RewardModel
from rlhf.nn.device import device_manager
from libs.utils.rouge_calculator import rouge_n_sentence_level

logger = logging.getLogger(__name__)
punc_patt = re.compile(f"[{re.escape(string.punctuation)}]")


def remove_punctuation(text: Text) -> Text:
    text = punc_patt.sub(" ", text)
    return text


class Rouge1F1Reward(RewardModel):
    def _cal_reward(self, doc: str, hyp: str, ref: str, *args, **kwargs):
        """Calculate Rouge-1 F1-score between the hypothesis and the reference summaries."""

        score = rouge_n_sentence_level(hyp=hyp, ref=ref, ns=[1])
        return score["rouge1"]["f"]

    def get_avg_reward(self):
        if self.n_samples == 0:
            return 0.0
        return self.total_reward / self.n_samples


class SentenceEmbeddingSimilarityReward(RewardModel):
    def __init__(self, sim_model, anchor_type: Literal["input", "output"]):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(sim_model)
        self.sim_model = AutoModel.from_pretrained(sim_model)
        self.sim_model.eval()
        self.anchor_type = anchor_type

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _cal_reward(self, doc: str, hyp: str, ref: str, training: bool = True, *args, **kwargs):
        """Calculate cosine similarity between the hypothesis summary and
        (1) the reference summary or (2) the input document."""

        if self.anchor_type == "input":
            anchor = doc
        else:
            anchor = ref
        inputs = self.tokenizer([hyp, anchor], padding=True, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        with torch.no_grad():
            outputs = self.sim_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        embeddings = self.mean_pooling(outputs, attention_mask)
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        reward = torch.sum(embeddings[0] * embeddings[1]).item() * 5
        return reward


def seed_reward_model(
    ppo_trainer,
    reward_model: RewardModel,
    tokenizer,
    dataloader,
    generation_kwargs: Dict,
    input_name: str,
    output_name: str,
    num_cut: int = -1,
):
    device = device_manager.device
    logger.info("Iterating sample dataset to seed the reward model...")
    progress_bar = tqdm(
        total=min(len(dataloader), num_cut if num_cut > 0 else float("inf")),
        desc="Batch",
    )
    for idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"][0].to(device)
        hyp_ids = ppo_trainer.generate(
            input_ids, **generation_kwargs
        )
        hyp_ids = hyp_ids.squeeze()
        hyp = tokenizer.decode(
            hyp_ids,
            clean_up_tokenization_spaces=False,
            skip_special_tokens=True
        )
        reward_model.cal_reward(
            doc=batch[input_name][0], hyp=hyp, ref=batch[output_name][0]
        )
        progress_bar.update(1)
        if (idx + 1) == num_cut:
            break
