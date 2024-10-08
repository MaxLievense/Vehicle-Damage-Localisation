from __future__ import annotations

import warnings

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.data.data import Data
from src.utils.base import Base

warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")


class BaseTrainer(Base):
    """
    Abstract Trainer class. All trainers should inherit from this class.
    """

    def __init__(
        self,
        device: torch.device,
        callbacks: dict,
        model,
        data: Data,
        **cfg: dict,
    ) -> None:
        super().__init__(cfg)

        self.log.debug("Building trainer...")
        self.device = device
        self.model = model
        self.data = data
        self.step = 0
        self.last_epoch = -1
        self.init_optimizer()
        self.callback = instantiate(DictConfig(callbacks))

        self.log.info("Loaded trainer.")

    def init_optimizer(self):
        self.optimizer = instantiate(DictConfig(self.cfg.optimizer), params=self.model.network.parameters())
        self.scheduler = (
            instantiate(DictConfig(self.cfg.scheduler), optimizer=self.optimizer) if self.cfg.get("scheduler") else None
        )

    def run(self):
        self.log.info("Running trainer...")
        self.train()

        if self.cfg.eval_at_end:
            self.eval()
        if self.cfg.save_last:
            self.save("end")

    def train(self):
        raise NotImplementedError

    def eval(self, loader="test"):
        raise NotImplementedError

    def save(self, tag=None):
        filename = f"{self.output_dir}/{self.last_epoch}{f'.{tag}' if tag else ''}.pth"
        state_dict = self.model.state_dict()
        torch.save(state_dict, filename)
        self.log.info(f"Model saved as {filename}")
