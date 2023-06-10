from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def get_pegasus_tokenizer_class():
    from transformers import PegasusTokenizer
    return PegasusTokenizer


def get_bart_tokenizer_class():
    from transformers import BartTokenizer
    return BartTokenizer


TOKENIZER_CLASS_MAPPING = {
    "AutoTokenizer": AutoTokenizer,
    "PegasusTokenizer": get_pegasus_tokenizer_class(),
    "BartTokenizer": get_bart_tokenizer_class()
}


def resolve_tokenizer_class(class_name):
    if class_name not in TOKENIZER_CLASS_MAPPING:
        raise Exception("Unknown tokenizer class: '{}'".format(class_name))
    return TOKENIZER_CLASS_MAPPING[class_name]


def get_pegasus_model_class():
    from transformers.models.pegasus.modeling_pegasus import PegasusForConditionalGeneration
    return PegasusForConditionalGeneration


def get_bart_model_class():
    from transformers.models.bart.modeling_bart import BartForConditionalGeneration
    return BartForConditionalGeneration


MODEL_CLASS_MAPPING = {
    "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
    "PegasusForConditionalGeneration": get_pegasus_model_class(),
    "BartForConditionalGeneration": get_bart_model_class()
}


def resolve_model_class(class_name):
    if class_name not in MODEL_CLASS_MAPPING:
        raise Exception("Unknown model class: '{}'".format(class_name))
    return MODEL_CLASS_MAPPING[class_name]
