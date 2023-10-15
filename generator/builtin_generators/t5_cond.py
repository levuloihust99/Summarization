import torch

from typing import Text, List, Union, Optional, Any, Dict
from transformers import AutoTokenizer, T5ForConditionalGeneration

from .common import greedy


class T5ConditionalGeneratorSummarizer:
    def __init__(self, model_path):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
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
        with torch.no_grad():
            if encoder_hidden_states is None:
                encoder_outputs = self.model.encoder(
                    input_ids=encoder_input_ids,
                    attention_mask=encoder_attention_mask,
                    return_dict=True
                )
                encoder_hidden_states = encoder_outputs.last_hidden_state

            decoder_outputs = self.model.decoder(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict=True
            )

            past_key_values = decoder_outputs.past_key_values
            sequence_output = decoder_outputs.last_hidden_state
            if self.model.config.tie_word_embeddings:
                # Rescale output before projecting on vocab
                # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
                sequence_output = sequence_output * (self.model.model_dim**-0.5)
            logits = self.model.lm_head(sequence_output) # [bsz, seq_len, vocab_size]
            next_token_logits = logits[:, -1:, :]
            tracker.update(
                past_key_values=past_key_values, encoder_hidden_states=encoder_hidden_states
            )
        return next_token_logits
    
    def greedy(self, inputs: Union[Text, List[Text]], max_length: int = 300) -> List[Text]:
        inputs = self.tokenizer(inputs, padding=True, return_tensors="pt")
        batch_size = inputs.input_ids.size(0)
        alive_seq = torch.full(
            [batch_size, 1],
            self.model.config.decoder_start_token_id,
            dtype=torch.long,
            device=self.device
        )
        outputs = greedy(
            self,
            encoder_input_ids=inputs.input_ids,
            encoder_attention_mask=inputs.attention_mask,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
            decoder_end_token_id=self.tokenizer.eos_token_id,
            decoder_input_ids=alive_seq,
            max_length=max_length
        )
        outputs = [
            self.tokenizer.decode(output, clean_up_tokenization_spaces=False, skip_special_tokens=True)
            for output in outputs
        ]
        return outputs
