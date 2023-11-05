import copy
from transformers import T5ForConditionalGeneration, T5Config

t5_default_config = {
    "d_ff": 3072,
    "d_kv": 64,
    "d_model": 768,
    "decoder_start_token_id": 0,
    "dropout_rate": 0.1,
    "eos_token_id": 1,
    "initializer_factor": 1.0,
    "is_encoder_decoder": True,
    "layer_norm_epsilon": 1e-06,
    "model_type": "t5",
    "n_positions": 512,
    "num_heads": 12,
    "num_layers": 12,
    "output_past": True,
    "pad_token_id": 0,
    "relative_attention_num_buckets": 32,
}


def init_model(
    model_type,
    decoder_start_token_id: int,
    pad_token_id: int,
    eos_token_id: int,
    vocab_size: int
):
    if model_type == "t5":
        cfg = copy.copy(t5_default_config)
        cfg.update(
            decoder_start_token_id=decoder_start_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            vocab_size=vocab_size,
        )
        t5_config = T5Config.from_dict(cfg)
        model = T5ForConditionalGeneration(t5_config)
        return model
    else:
        raise Exception("Unsupported model type: '{}'".format(model_type))
