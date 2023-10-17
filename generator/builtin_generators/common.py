import torch

from typing import Optional, Literal
from tqdm import tqdm

from ..utils import recursive_apply


def greedy(
    model,
    encoder_input_ids: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    decoder_start_token_id: int,
    decoder_end_token_id: int,
    decoder_input_ids: Optional[torch.Tensor] = None,
    max_length: int = 100
):
    device = encoder_input_ids.device
    batch_size = encoder_input_ids.size(0)
    outputs = [None] * batch_size
    if decoder_input_ids is not None:
        alive_seq = decoder_input_ids
    else:
        alive_seq = torch.full(
            [batch_size, 1],
            decoder_start_token_id,
            dtype=torch.long,
            device=device
        )
    generated_tokens = 0
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)
    tracker = {
        "encoder_input_ids": encoder_input_ids,
        "encoder_attention_mask": encoder_attention_mask,
        "input_ids": alive_seq
    } # store model-specific caching
    progress_bar = tqdm(total=max_length)
    while True:
        params = tracker
        tracker = {}
        logits = model.forward(tracker, **params)
        next_ids = torch.argmax(logits, dim=-1) # [bsz, 1]
        tracker.update(input_ids=next_ids)
        alive_seq = torch.cat([alive_seq, next_ids], dim=1) # [bsz, seq_len]
        is_finished = (next_ids == decoder_end_token_id).view(-1)
        for idx in range(is_finished.size(0)):
            if is_finished[idx]:
                outputs[batch_indices[idx]] = alive_seq[idx]
        selected_indices = is_finished.eq(0).nonzero().view(-1)
        if selected_indices.size(0) < alive_seq.size(0):
            if selected_indices.size(0) == 0:
                break
            alive_seq = alive_seq.index_select(0, selected_indices)
            batch_indices = batch_indices.index_select(0, selected_indices)
            tracker.update(
                input_ids=tracker["input_ids"].index_select(0, selected_indices),
                past_key_values=recursive_apply(tracker["past_key_values"], fn=lambda x: x.index_select(0, selected_indices)),
                encoder_hidden_states=tracker["encoder_hidden_states"].index_select(0, selected_indices),
                encoder_attention_mask=tracker["encoder_attention_mask"].index_select(0, selected_indices)
            )

        generated_tokens += 1
        progress_bar.update(1)
        if generated_tokens == max_length:
            break
    
    if selected_indices.size(0) > 0:
        for idx in selected_indices:
            outputs[batch_indices[idx]] = alive_seq[idx]

    return outputs
