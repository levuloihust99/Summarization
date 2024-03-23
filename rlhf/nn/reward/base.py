from abc import ABC, abstractmethod


class RewardModel(ABC):
    def __init__(self):
        self.total_reward = 0.0
        self.n_samples = 0

    def cal_reward(self, doc: str, hyp: str, ref: str, training: bool = True, *args, **kwargs):
        """Calculate reward based on: document, hypothesis, reference."""
        r = self._cal_reward(doc, hyp, ref)
        if training:
            self.total_reward += r
            self.n_samples += 1
        return r

    @abstractmethod
    def _cal_reward(self, doc: str, hyp: str, ref: str, *args, **kwargs):
        """Calculate reward based on: document, hypothesis, reference."""

    def get_avg_reward(self):
        if self.n_samples == 0:
            return 0.0
        return self.total_reward / self.n_samples
