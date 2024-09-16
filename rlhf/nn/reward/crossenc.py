import torch
from torch import nn

from transformers import AutoTokenizer
from transformers.models.t5.modeling_t5 import T5Stack, T5ForConditionalGeneration

from .base import RewardModel
from rlhf.nn.device import device_manager


class CrossEncoder(nn.Module):
    """Take a pair of input texts, return a scalar value which measure the similarity between the text pair.
    This class should take any model architecture, e.g. T5, GPT, BERT, .etc"""


class T5CrossEncoder(CrossEncoder, T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)
        self.score = nn.Linear(config.hidden_size, 1, bias=False)

    @classmethod
    def from_t5_for_conditional_generation(cls, encoder_decoder_model):
        encoder_stack_config = encoder_decoder_model.config
        encoder_stack_config.is_decoder = False
        encoder_stack_config.use_cache = False
        encoder_stack_config.is_encoder_decoder = False
        embeddings = encoder_decoder_model.shared
        model = T5CrossEncoder(encoder_stack_config, embeddings)
        model_state_dict = {}
        p_names = []
        for p, value in encoder_decoder_model.named_parameters():
            p_names.append(p)
            if p.startswith('encoder.'):
                model_state_dict[p[len('encoder.'):]] = value
            elif p == 'shared.weight':
                model_state_dict['embed_tokens.weight'] = value
        model_state_dict['score.weight'] = model.score.weight
        model.load_state_dict(model_state_dict)
        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        t5stack_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        seq_output = t5stack_outputs.last_hidden_state # [bsz, seq_len, hidden_size]
        score = self.score(seq_output) # [bsz, seq_len, 1]
        score = torch.squeeze(score, dim=-1) # [bsz, seq_len]

        if input_ids is not None:
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
        else:
            batch_size = inputs_embeds.shape[0]
            seq_len = inputs_embeds.shape[1]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % seq_len
                sequence_lengths = sequence_lengths.to(score.device)
            else:
                sequence_lengths = -1

        pooled_score = score[torch.arange(batch_size, device=score.device), sequence_lengths] # [bsz, 1]
        return pooled_score


class T5CrossEncoderReward(RewardModel):
    def __init__(self, ckpt_path: str, pretrained_path: str, sep_token: str):
        super().__init__()
        self.sep_token = sep_token
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        saved_state = torch.load(ckpt_path, map_location=lambda s, t: s)
        pretrained = T5ForConditionalGeneration.from_pretrained(pretrained_path)
        cross_enc = T5CrossEncoder.from_t5_for_conditional_generation(pretrained)
        cross_enc.load_state_dict(saved_state)
        self.cross_enc = cross_enc.eval().to(device_manager.device)

    def _cal_reward(self, doc: str, hyp: str, ref: str, *args, **kwargs):
        doc_tokens = self.tokenizer.tokenize(doc)
        doc_ids = self.tokenizer.convert_tokens_to_ids(doc_tokens)
        hyp_tokens = self.tokenizer.tokenize(hyp)
        hyp_ids = self.tokenizer.convert_tokens_to_ids(hyp_tokens)
        input_ids = (
            doc_ids
            + self.tokenizer.convert_tokens_to_ids([self.sep_token])
            + hyp_ids
            + [self.tokenizer.eos_token_id]
        )
        input_ids = torch.tensor([input_ids]).to(device_manager.device)
        with torch.no_grad():
            score = self.cross_enc(input_ids=input_ids)
        score = score[0].item()
        self.n_samples += 1
        self.total_reward += score
        return score
