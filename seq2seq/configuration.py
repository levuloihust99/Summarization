import logging

logger = logging.getLogger(__name__)


class Seq2SeqConfig:
    def __init__(self, **kwargs):
        # tokenizer
        self.tokenizer_path = "levuloihust/vien-unigram-tokenizer"
        self.use_fast = False

        # model
        self.model_type = "t5"
        self.model_path = None

        # dataset
        self.train_data_path = None
        self.valid_data_path = None

        # training config
        self.output_dir = "assets/outputs"
        self.do_train = True
        self.do_eval = True
        self.learning_rate = 4e-3
        self.num_train_epochs = 10
        self.warmup_ratio = 0.0
        self.warmup_steps = 0
        self.weight_decay = 0
        self.per_device_train_batch_size = 2
        self.per_device_eval_batch_size = 2
        self.logging_dir = "assets/logs"
        self.group_by_length = False
        self.save_strategy = "steps"
        self.save_steps = 100
        self.evaluation_strategy = "steps"
        self.eval_steps = 100
        self.save_total_limit = 2
        self.fp16 = False
        self.gradient_accumulation_steps = 1
        self.gradient_checkpointing = False
        self.log_level = "info"
        self.logging_steps = 50
        self.logging_first_step = True
        self.max_grad_norm = 2.0
        self.label_smoothing_factor = 0.1
        self.report_to = ["tensorboard"]
        self.load_best_model_at_end = True
        self.metric_for_best_model = "acc"
        self.greater_is_better = True
        self.predict_with_generate = False
        self.resume_from_checkpoint = None
        self.remove_unused_columns = False
        self.generation_max_length = 100
        self.generation_num_beams = 1
        self.input_name = "prompt"
        self.output_name = "completion"
        self.data_seed = None
        self.max_input_len = None
        self.max_output_len = None

        # data config
        self.input_transform = None # [json_sequentialize]
        self.output_transform = None # [json_sequentialize]
        
        self.add_run_id = False
        self.wandb_api_key = None

        self.override_defaults(**kwargs)
        self.validate_config()

    def override_defaults(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.__dict__:
                logger.warn("Unknown hparam " + k)
            self.__dict__[k] = v

    def validate_config(self):
        pass
    
    def to_json(self):
        return {
            "tokenizer_path": self.tokenizer_path,
            "use_fast": self.use_fast,
            "model_type": self.model_type,
            "model_path": self.model_path,
            "train_data_path": self.train_data_path,
            "valid_data_path": self.valid_data_path,
            "output_dir": self.output_dir,
            "do_train": self.do_train,
            "do_eval": self.do_eval,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "warmup_ratio": self.warmup_ratio,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "logging_dir": self.logging_dir,
            "group_by_length": self.group_by_length,
            "save_strategy": self.save_strategy,
            "save_steps": self.save_steps,
            "evaluation_strategy": self.evaluation_strategy,
            "eval_steps": self.eval_steps,
            "save_total_limit": self.save_total_limit,
            "fp16": self.fp16,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_checkpointing": self.gradient_checkpointing,
            "log_level": self.log_level,
            "logging_steps": self.logging_steps,
            "logging_first_step": self.logging_first_step,
            "max_grad_norm": self.max_grad_norm,
            "label_smoothing_factor": self.label_smoothing_factor,
            "report_to": self.report_to,
            "load_best_model_at_end": self.load_best_model_at_end,
            "metric_for_best_model": self.metric_for_best_model,
            "greater_is_better": self.greater_is_better,
            "predict_with_generate": self.predict_with_generate,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "generation_max_length": self.generation_max_length,
            "generation_num_beams": self.generation_num_beams,
            "input_name": self.input_name,
            "output_name": self.output_name,
            "data_seed": self.data_seed,
            "max_input_len": self.max_input_len,
            "max_output_len": self.max_output_len,
            "input_transform": self.input_transform,
            "output_transform": self.output_transform,
            "add_run_id": self.add_run_id,
            "wandb_api_key": self.wandb_api_key,
        }
