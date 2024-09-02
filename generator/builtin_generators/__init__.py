from typing import Literal
from .t5_cond import T5ConditionalGeneratorSummarizer
from .mbart_cond import MBartConditionalGeneratorSummarizer


def resolve_summarizer_class(
    model_type: Literal["t5_cond", "mbart_cond"]
):
    if model_type == "t5_cond":
        return T5ConditionalGeneratorSummarizer
    if model_type == "mbart_cond":
        return MBartConditionalGeneratorSummarizer
