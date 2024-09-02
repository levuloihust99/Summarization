import torch

from typing import Text, List, Union, Optional, Any, Dict
from transformers import AutoTokenizer, MBartForConditionalGeneration

from .common import greedy


class MBartConditionalGeneratorSummarizer:
    def __init__(
        self,
        model_path,
        tokenizer_path: Optional[Text] = None,
        max_input_len: Optional[int] = None,
    ):
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.device = device
        self.model = MBartForConditionalGeneration.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
        if tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_input_len = max_input_len

    def forward(
        self,
        tracker: Dict[Text, Any],
        encoder_input_ids: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values = None,
        **kwargs
    ):
        encoder = self.model.get_encoder()
        decoder = self.model.get_decoder()
        with torch.no_grad():
            if encoder_hidden_states is None:
                encoder_outputs = encoder(
                    input_ids=encoder_input_ids,
                    attention_mask=encoder_attention_mask,
                    return_dict=True
                )
                encoder_hidden_states = encoder_outputs.last_hidden_state

            decoder_outputs = decoder(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict=True
            )

            past_key_values = decoder_outputs.past_key_values
            sequence_output = decoder_outputs.last_hidden_state
            logits = self.model.lm_head(sequence_output) + self.model.final_logits_bias # [bsz, seq_len, vocab_size]
            next_token_logits = logits[:, -1:, :]
            tracker.update(
                past_key_values=past_key_values,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask
            )
        return next_token_logits

    def greedy(
        self,
        inputs: Union[Text, List[Text]],
        max_length: int = 300,
        block_n_grams: int = -1,
        **kwargs
    ) -> List[Text]:
        inputs = self.tokenizer(
            inputs,
            padding=True,
            max_length=self.max_input_len or self.model.config.max_position_embeddings,
            truncation=True,
            return_tensors="pt"
        )
        batch_size = inputs.input_ids.size(0)
        alive_seq = torch.full(
            [batch_size, 1],
            self.model.config.decoder_start_token_id,
            dtype=torch.long,
            device=self.device
        )
        outputs = greedy(
            self,
            encoder_input_ids=inputs.input_ids.to(self.device),
            encoder_attention_mask=inputs.attention_mask.to(self.device),
            decoder_start_token_id=self.model.config.decoder_start_token_id,
            decoder_end_token_id=self.tokenizer.eos_token_id,
            decoder_input_ids=alive_seq,
            max_length=max_length,
            block_n_grams=block_n_grams,
            **kwargs,
        )
        outputs = [
            self.tokenizer.decode(output, clean_up_tokenization_spaces=False, skip_special_tokens=True)
            for output in outputs
        ]
        return outputs
