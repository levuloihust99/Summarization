import logging
import dataclasses

from rlhf.data_helpers.dataset_config import DatasetConfig

logger = logging.getLogger(__name__)


class RLHFTrainingConfig:
    attrs = set()

    def __init__(self, **kwargs):
        # training
        self.cpu = False
        self.batch_size = 2
        self.model_save_path = "outputs"
        self.save_steps = 100
        self.do_eval = False
        self.logging_dir = "logs"
        self.log_level = "info"
        self.log_with = "wandb"
        self.logging_steps = 10
        self.wandb_api_key = None
        self.keep_checkpoint_max = 3
        self.eval_on_first_step = False
        self.learning_rate = 1e-5
        self.num_train_epochs = 3
        self.resume_from_checkpoint = None
        self.eval_mode = "reward"

        # tokenizer
        self.tokenizer_path = "Yale-LILY/brio-xsum-cased"
        self.tokenizer_class = "PegasusTokenizer"

        # model
        self.model_path = "Yale-LILY/brio-xsum-cased"
        self.model_class = "PegasusForConditionalGeneration"

        # data
        self.data_name = "vietnews"
        self.dataset_config: DatasetConfig = None
        self.train_data_path = None # must be provided
        self.eval_data_path = None

        # others
        self.seed = 12345
        self.reward_model = "rouge1-f1"
        self.greater_is_better = True
        self.metric_for_best_model = "eval/rouge1-f1"
        self.baseline = "zero"
        self.sim_model = "NtDNlp/sentence-embedding-vietnamese"
        self.crossenc_ckpt_path = None
        self.crossenc_sep_token = "<extra_id_0>"
        self.crossenc_pretrained = "VietAI/vit5-base"
        self.input_name = "document"
        self.output_name = "summary"
        self.seed_reward_data_cut = 100
        self.anchor = "input"

        self.override_defaults(**kwargs)

    def override_defaults(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.__dict__:
                logger.warn("Unknown hparam " + k)
            self.__dict__[k] = v

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v

    def __setattr__(self, name, value):
        cls = type(self)
        if name not in cls.attrs:
            cls.attrs.add(name)
        super().__setattr__(name, value)

    def to_json(self):
        json_obj = {}
        for attr in type(self).attrs:
            value = getattr(self, attr)
            if dataclasses.is_dataclass(value):
                value = dataclasses.asdict(value)
            json_obj[attr] = value
        return json_obj
