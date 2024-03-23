from typing import Union
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    name: str
    max_len: int
    total_len: Union[int] = None


vietnews_config = DatasetConfig(name="vietnews", max_len=200)
xsum_config = DatasetConfig(name="xsum", max_len=80, total_len=512)
cnndm_config = DatasetConfig(name="cnndm", max_len=120, total_len=1024)


DATASET_CONFIGS = {
    "vietnews": vietnews_config,
    "xsum": xsum_config,
    "cnndm": cnndm_config
}
