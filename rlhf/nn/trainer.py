import os
import json
import torch
import shutil
import logging
import warnings

from tqdm import tqdm
from typing import Optional, Dict, List, Text, Iterable
from trl import PPOTrainer
from torch.utils.data import DataLoader
from accelerate.utils import gather_object

from libs.utils.rng import get_rng_state, set_rng_state
from libs.data_helpers.utils import ByteDatasetWriter
from libs.data_helpers.bytedataset import ByteDataset
from rlhf.nn.reward.base import RewardModel
from rlhf.nn.reward.common import seed_reward_model
from rlhf.nn.device import device_manager
from rlhf.nn.checkpoint import TrainingState, RNGState
from rlhf.nn.configuration import RLHFTrainingConfig
from rlhf.data_helpers.dataloader import get_data_collator
from rlhf.nn.evaluation import evaluate_generator

logger = logging.getLogger(__name__)


def get_metric(valid_stats, metric_for_best_model):
    if metric_for_best_model not in valid_stats:
        avail_stats = list(valid_stats.keys())
        avail_stats = ", ".join(valid_stats)
        raise Exception(
            "Metric {} is not in the eval stats. Available stat are: {}.".format(
                avail_stats
            )
        )
    return valid_stats[metric_for_best_model]


class RLHFTrainer:
    def __init__(
        self,
        config: RLHFTrainingConfig,
        ppo_trainer: PPOTrainer,
        tokenizer,
        train_dataloader,
        eval_dataloader,
        reward_model: RewardModel,
        training_state: Optional[TrainingState],
        rng_state: RNGState,
        run_id: str
    ):
        self.config = config
        self.ppo_trainer = ppo_trainer
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.reward_model = reward_model
        self.training_state = training_state
        self.rng_state = rng_state
        self.run_id = run_id

    def train(self):
        # generation config
        generation_kwargs = {
            "min_length": 4,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": self.config.dataset_config.max_len,
            "no_repeat_ngram_size": 3,
        }
        eval_generation_kwargs = {
            "no_repeat_ngram_size": 3,
            "max_new_tokens": self.config.dataset_config.max_len,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if self.config.baseline == "avg":
            if self.training_state is None:
                sample_dataset = ByteDataset(
                    data_path=self.config.valid_data_path, idx_record_size=6
                )
                sample_dataloader = DataLoader(
                    sample_dataset,
                    batch_size=1,
                    shuffle=False,
                    collate_fn=get_data_collator(
                        self.tokenizer,
                        self.config.input_name,
                        self.config.output_name,
                        max_input_len=self.config.dataset_config.total_len,
                        max_output_len=self.config.dataset_config.max_len,
                    ),
                )
                seed_reward_model(
                    ppo_trainer=self.ppo_trainer,
                    reward_model=self.reward_model,
                    tokenizer=self.tokenizer,
                    dataloader=sample_dataloader,
                    generation_kwargs=eval_generation_kwargs,
                    input_name=self.config.input_name,
                    output_name=self.config.output_name,
                    num_cut=self.config.seed_reward_data_cut
                ) # change torch RNG
            else:
                self.reward_model.total_reward = self.training_state.reward_state.total_reward
                self.reward_model.n_samples = self.training_state.reward_state.n_samples

        # past training state
        best_metric = float("-inf")
        best_checkpoint = None
        trained_epochs = 0
        data_step = 0
        global_step = 0
        if self.training_state:
            best_metric = self.training_state.best_metric
            best_checkpoint = self.training_state.best_checkpoint
            trained_epochs = self.training_state.epoch
            data_step = self.training_state.data_step
            global_step = self.training_state.global_step
        self.ppo_trainer.current_step = global_step

        if (
            self.config.do_eval
            and self.config.eval_on_first_step
            and self.training_state is None
        ):
            eval_stats = evaluate_generator(
                model=self.ppo_trainer.accelerator.unwrap_model(
                    self.ppo_trainer.model
                ).pretrained_model,
                dataloader=self.eval_dataloader,
                tokenizer=self.tokenizer,
                input_name=self.config.input_name,
                output_name=self.config.output_name,
                generation_kwargs=eval_generation_kwargs,
                eval_mode=self.config.eval_mode,
                reward_model=self.reward_model if self.config.eval_mode == "reward" else None,
            ) # change torch RNG state
            self.ppo_trainer.accelerator.log(eval_stats, step=global_step)

        total_steps = len(self.train_dataloader) * self.config.num_train_epochs
        progress_bar = tqdm(desc="Step", total=total_steps, initial=global_step)
        self.ppo_trainer.model.train()

        for epoch in range(trained_epochs, self.config.num_train_epochs):
            if data_step == len(self.train_dataloader):
                data_step = 0
                set_rng_state(self.rng_state)
                continue

            self.train_dataloader.sampler.set_epoch(epoch)
            data_iterator = iter(self.train_dataloader) # change RNG state
            for i in range(data_step):
                next(data_iterator) # may change RNG state
            if data_step > 0: # this is e resume
                set_rng_state(self.rng_state)

            for i, batch in enumerate(data_iterator):
                step = i + data_step
                query_tensors = batch["input_ids"]
                query_tensors = [q.to(device_manager.device) for q in query_tensors]
                response_tensors = []
                response_texts = []
                rewards = []
                ignored_idxs = set() # PPO raise ValueError when repsonse tensor is less then 4 tokens

                for _i, query in enumerate(query_tensors):
                    # response = self.ppo_trainer.generate(query, **generation_kwargs)
                    response = self.ppo_trainer.model.pretrained_model.generate(torch.stack([query], dim=0), **generation_kwargs)
                    response = response.squeeze()
                    if len(response) < 4:
                        ignored_idxs.add(_i)
                        continue

                    response_tensors.append(response)
                    response_text = self.tokenizer.decode(
                        response,
                        clean_up_tokenization_spaces=False,
                        skip_special_tokens=True
                    )
                    response_texts.append(response_text)

                    # calculate reward
                    if self.config.baseline == "avg":
                        avg_reward = self.reward_model.get_avg_reward()
                    else:
                        avg_reward = 0.0
                    reward = self.reward_model.cal_reward(
                        doc=batch[self.config.input_name][_i],
                        hyp=response_text,
                        ref=batch[self.config.output_name][_i],
                    )
                    reward = reward - avg_reward
                    rewards.append(torch.tensor(reward).to(device_manager.device))

                # filter query_tensors
                included_idxs = []
                filtered_query_tensors = []
                for _i in range(len(query_tensors)):
                    included_idxs.append(_i)
                    filtered_query_tensors.append(query_tensors[_i])
                if not filtered_query_tensors:
                    continue
                actual_bsz = len(filtered_query_tensors)

                # PPO step
                if (global_step + 1) % self.config.logging_steps == 0:
                    train_stats = self.ppo_trainer.step(filtered_query_tensors, response_tensors, rewards)
                    batch_to_log = {
                        "epoch": [epoch] * actual_bsz,
                        "step": [step] * actual_bsz,
                        "global_step": [global_step] * actual_bsz,
                        "document": batch[self.config.input_name],
                        "reference": batch[self.config.output_name],
                        "hypothesis": response_texts,
                        "reward": rewards
                    }
                    self.log_stats(train_stats)
                    self.log_batch(batch_to_log)
                    self.trl_log_stats(
                        current_step=global_step + 1,
                        stats=train_stats,
                        batch=batch_to_log,
                        rewards=rewards,
                        columns_to_log=list(batch_to_log.keys()),
                    )

                if (global_step + 1) % self.config.save_steps == 0:
                    # evaluation
                    if self.config.do_eval:
                        eval_stats = evaluate_generator(
                            model=self.ppo_trainer.accelerator.unwrap_model(
                                self.ppo_trainer.model
                            ).pretrained_model,
                            dataloader=self.eval_dataloader,
                            tokenizer=self.tokenizer,
                            output_name=self.config.output_name,
                            input_name=self.config.input_name,
                            generation_kwargs=eval_generation_kwargs,
                            eval_mode=self.config.eval_mode,
                            reward_model=self.reward_model if self.config.eval_mode == "reward" else None,
                        )
                        self.ppo_trainer.accelerator.log(eval_stats, step=global_step + 1)

                    if self.ppo_trainer.accelerator.is_main_process:
                        cp_name = f"checkpoint-{global_step + 1}"
                        cp_path = os.path.join(self.config.model_save_path, self.run_id, cp_name)

                        if self.config.do_eval:
                            metric = get_metric(eval_stats, self.config.metric_for_best_model)
                            if not self.config.greater_is_better:
                                metric = -metric
                            if metric > best_metric:
                                best_metric = metric
                                best_checkpoint = cp_name
                                logger.info("New best checkpoint: {}".format(cp_name))

                        training_state = {
                            "epoch": epoch,
                            "global_step": global_step + 1,
                            "data_step": step + 1,
                            "best_checkpoint": best_checkpoint,
                            "best_metric": best_metric * int(self.config.greater_is_better)
                        }
                        if self.config.baseline == "avg":
                            training_state["reward_model"] = {
                                "total_reward": self.reward_model.total_reward,
                                "n_samples": self.reward_model.n_samples
                            }

                        self.save_checkpoint(cp_path, training_state)

                global_step += 1
                progress_bar.update(1)

            data_step = 0

    def save_checkpoint(self, cp_path, training_state):
        cp_dir = os.path.join(self.config.model_save_path, self.run_id)
        if not os.path.exists(cp_dir):
            os.makedirs(cp_dir)

        # checkpoints to be deleted if necessary
        all_cps = os.listdir(cp_dir)
        all_cps = [f for f in all_cps if f.startswith('checkpoint-')]
        all_cps = [os.path.join(cp_dir, f) for f in all_cps]
        all_cps = sorted(all_cps, key=lambda x: os.stat(x).st_mtime, reverse=True)
        if self.config.do_eval:
            best_cp = []
            other_cps = []
            for cp in all_cps:
                if cp == os.path.join(cp_dir, training_state["best_checkpoint"]):
                    best_cp.append(cp)
                else:
                    other_cps.append(cp)
            if len(best_cp) == 1 and best_cp[0] == cp_path:
                head = [cp_path]
            else:
                head = best_cp + [cp_path]
            all_cps = head + other_cps
            visited = set()
            non_duplicated_all_cps = []
            for cp in all_cps:
                if cp not in visited:
                    visited.add(cp)
                    non_duplicated_all_cps.append(cp)
            all_cps = non_duplicated_all_cps
        else:
            all_cps = [cp_path] + all_cps

        cps_to_delete = []
        if self.config.keep_checkpoint_max > 0:
            cps_to_delete = all_cps[self.config.keep_checkpoint_max:]

        # create a directory for saving checkpoint
        if not os.path.exists(cp_path):
            os.makedirs(cp_path)

        # save tokenizer
        self.tokenizer.save_pretrained(cp_path)
        logger.info("Saved tokenizer into '{}'".format(cp_path))

        # save model
        logger.info("Saving model...")
        self.ppo_trainer.accelerator.unwrap_model(
            self.ppo_trainer.model
        ).save_pretrained(cp_path)
        logger.info("Saved model into '{}'".format(cp_path))

        # save optimizer
        logger.info("Saving optimizer state...")
        optimizer_save_file = os.path.join(cp_path, "optimizer.pt")
        torch.save(self.ppo_trainer.optimizer.state_dict(), optimizer_save_file)
        logger.info("Saved optimizer state into '{}'".format(optimizer_save_file))

        # save RNG state
        rng_state = get_rng_state()
        rng_state_file = os.path.join(cp_path, "rng_state.pth")
        torch.save(rng_state, rng_state_file)
        logger.info("Saved RNG states into '{}'".format(rng_state_file))

        # save training state
        training_state_file = os.path.join(cp_path, "training_state.json")
        with open(training_state_file, "w") as writer:
            json.dump(training_state, writer, indent=4, ensure_ascii=False)
        logger.info("Saved training state into '{}'".format(training_state_file))

        # delete old checkpoints if necessary
        for cp in cps_to_delete:
            logger.info(
                "Deleting {} since maximum kept checkpoint is {}...".format(
                    cp, self.config.keep_checkpoint_max
                )
            )
            shutil.rmtree(cp)

    def log_batch(self, batch_to_log: Dict[Text, List]):
        if not batch_to_log:
            return

        log_dir = os.path.join(self.config.logging_dir, self.run_id)
        table_dir = os.path.join(log_dir, "table")
        if not os.path.exists(table_dir):
            os.makedirs(table_dir)

        writer = ByteDatasetWriter(table_dir)
        keys = batch_to_log.keys()
        data = zip(*batch_to_log.values())
        for item in data:
            dict_item = dict(zip(keys, item))
            writer.add_item(dict_item)

    def log_stats(self, stats: Dict):
        if not stats:
            return

        log_dir = os.path.join(self.config.logging_dir, self.run_id)
        stats_dir = os.path.join(log_dir, "stats")
        if not os.path.exists(stats_dir):
            os.makedirs(stats_dir)

        writer = ByteDatasetWriter(stats_dir)
        writer.add_item(stats)

    def trl_log_stats(
        self,
        current_step: int,
        stats: dict,
        batch: dict,
        rewards: List[torch.FloatTensor],
        columns_to_log: Iterable[str] = ("query", "response"),
    ):
        """
        This code is borrowed from `trl/trainer/ppo_trainer.py`. I need to modify the code to suit step controlling.

        A function that logs all the training stats. Call it at the end of each epoch.

        Args:
            stats (dict[str, Any]):
                A dictionary of training stats.
            batch (dict[str, Any]):
                A dictionary of batch data, this contains the queries and responses.
            rewards (`List[torch.FloatTensor]`):
                A tensor of rewards.
        """
        ppo_trainer = self.ppo_trainer
        # all gather stats
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards).to(ppo_trainer.current_device)
        rewards = ppo_trainer.accelerator.gather(rewards).flatten()

        if ppo_trainer.config.log_with == "wandb":
            import wandb

            if any(column_to_log not in batch.keys() for column_to_log in columns_to_log):
                raise ValueError(f"Columns to log {columns_to_log} are not present in the batch {batch.keys()}.")

            batch_list = [batch[column_to_log] for column_to_log in columns_to_log]
            if ppo_trainer.is_distributed:
                gathered_batch_list = []
                for b in batch_list:
                    flattened = gather_object(b)
                    gathered_batch_list.append(flattened)
                batch_list = gathered_batch_list

        # Log only if we are in the main process
        if ppo_trainer.accelerator.is_main_process:
            logs = {}

            # Log stats
            if "query" not in batch.keys() and "response" not in batch.keys():
                # warn the user that the game logs will not be logged
                warnings.warn(
                    "The game logs will not be logged because the batch does not contain the keys 'query' and "
                    "'response'. "
                )
            elif ppo_trainer.config.log_with == "wandb":
                table_rows = [list(r) for r in zip(*batch_list, rewards.cpu().tolist())]
                logs.update({"game_log": wandb.Table(columns=[*columns_to_log, "reward"], rows=table_rows)})

            logs.update(stats)

            # manually cast in fp32 for bf16 torch tensors
            for k, v in logs.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                    logs[k] = v.float()

            logs["env/reward_mean"] = torch.mean(rewards).cpu().numpy().item()
            logs["env/reward_std"] = torch.std(rewards).cpu().numpy().item()
            logs["env/reward_dist"] = rewards.cpu().numpy()

            # update the current step
            ppo_trainer.accelerator.log(
                logs,
                step=current_step,
            )
