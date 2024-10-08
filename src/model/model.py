from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torchvision
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.utils.base import Base

if TYPE_CHECKING:
    from src.data.data import Data


class Model(torch.nn.Module, Base):
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

    def _init_modules(self):
        self.network = instantiate(DictConfig(self.cfg.network), n_classes=self.n_classes, device=self.device)
        self.network.to(self.device)
        self.log.info(f"Loaded model:\n{self.network}")

        if self.cfg.get("criterion"):
            self.criterion = instantiate(DictConfig(self.cfg.criterion))
            self.criterion.to(self.device)
            self.log.info(f"Loaded criterion:\n{self.criterion}")
        else:
            self.criterion = None
            self.log.warning("No criterion loaded.")

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return self.criterion.post_forward(self.network(x))

    def step(self, trainer, data, target):
        if self.network.training:
            with torch.set_grad_enabled(True):
                print(target)
                output = self.network(data)
                loss = self.criterion(output, target)
                loss.backward()
                output = self.criterion.post_forward(output)
                return loss.item(), output
        return None, self.network(data)

    def pretrain(self, *args, **kwargs):
        pass

    def posttrain(self, *args, **kwargs):
        pass

    def preepoch(self, *args, **kwargs):
        torch.cuda.empty_cache()

    def postepoch(self, *args, **kwargs):
        pass


class FasterRCNN(Model):
    def _init_modules(self):
        super()._init_modules()
        in_features = self.network.roi_heads.box_predictor.cls_score.in_features
        self.network.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, self.n_classes + 1
        )  # add one for background
        self.network.to(self.device)

    def step(self, trainer, data, target):
        if self.network.training:
            with torch.set_grad_enabled(True):
                # TODO: The Dataloader returns tensors on CPU, trainer only sends to device if tensor, not dict.
                for i, _target in enumerate(target):
                    for k, v in _target.items():
                        target[i][k] = v.to(self.device)
                output = self.network(data, target)
                loss = sum([o for o in output.values()])
                loss.backward()

                if self.cfg.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.cfg.clip_grad)
                return loss.item(), output
        output = self.network(data)
        return None, output
