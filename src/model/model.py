from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.utils.base import Base

if TYPE_CHECKING:
    from src.data.data import Data


class BaseModel(torch.nn.Module, Base):
    """Abstract Model class. All models should inherit from this class."""

    def __init__(
        self,
        device: torch.device,
        data: Data,
        **cfg: dict,
    ) -> None:
        super().__init__()
        Base.__init__(self, cfg)

        self.log.debug("Building model...")
        self.device = device
        self.data = data
        self.n_classes = data.n_classes
        self._init_modules()
        self.log.info(f"Loaded model:\n{self.network}")

    def _init_modules(self):
        self.network = instantiate(DictConfig(self.cfg.network), n_classes=self.n_classes, device=self.device)
        self.network.to(self.device)

        if self.cfg.get("criterion"):
            self.criterion = instantiate(DictConfig(self.cfg.criterion))
            self.criterion.to(self.device)
            self.log.info(f"Loaded criterion:\n{self.criterion}")
        else:
            self.criterion = None
            self.log.warning("No criterion loaded. Assuming criterion is defined in network.")

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return self.network(x)

    def step(self, trainer, data, target):
        raise NotImplementedError

    def pretrain(self, *args, **kwargs):
        pass

    def posttrain(self, *args, **kwargs):
        pass

    def preepoch(self, *args, **kwargs):
        torch.cuda.empty_cache()

    def postepoch(self, *args, **kwargs):
        pass
