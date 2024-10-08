from __future__ import annotations

import warnings

import torch
import wandb
from ACIL.data.data import Data
from ACIL.utils.base import Base
from ACIL.utils.metrics import class_metrics
from hydra.utils import instantiate
from omegaconf import DictConfig

from model.model import Model

warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")


class Trainer(Base):
    def __init__(
        self,
        device: torch.device,
        callbacks: dict,
        model: Model,
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
        self.model.network.train()
        self.model.pretrain(self)

        self.log.info("Training...")
        for epoch in range(self.cfg.epochs):
            self.last_epoch = epoch
            _breaking = False
            self.model.preepoch(self)

            running_loss = 0.0
            running_accuracy = 0.0
            training_entries = 0
            for data in self.data.train_loader:
                training_entries += len(data[0])
                self.model.network.train()
                data = [_data.to(self.device) if isinstance(_data, torch.Tensor) else _data for _data in data]
                self.optimizer.zero_grad()
                step_loss, logits = self.model.step(self, *data)
                self.optimizer.step()
                running_loss += step_loss
                running_accuracy += (torch.argmax(logits, dim=1) == data[1]).to("cpu").sum().float().item()
            _lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]["lr"]
            self.log.info(
                f"Epoch {epoch+1}/{self.cfg.epochs} | "
                + f"avg loss: {running_loss / training_entries:.6f} "
                + f"(lr: {_lr:.5f}) | "
                + f"avg accuracy: {running_accuracy / training_entries:.2%}"
            )
            accuracy = running_accuracy / training_entries
            if self.callback(
                trainer=self,
                loss=running_loss,
                accuracy=accuracy,
                epoch=epoch,
                outputs=logits,
                data=data,
                lr=_lr,
            ):
                _breaking = True
            self.model.postepoch(self)

            if self.scheduler:
                self.scheduler.step(epoch)

            if _breaking:
                break
        self.model.posttrain(self)

    def eval(self, loader="test"):
        if loader == "test":
            self.log.info("Evaluating...")

        _loader = getattr(self.data, loader + "_loader")
        total_labels = []
        total_logits = []
        self.model.network.eval()
        with torch.no_grad():
            for _, data in enumerate(_loader):
                data = [_data.to(self.device) if isinstance(_data, torch.Tensor) else _data for _data in data]
                _, logits = self.model.step(self, *data)
                total_logits.append(logits)
                total_labels.append(data[1])
        closed_results, _ = class_metrics(
            y_true=torch.cat(total_labels),
            y_out=torch.cat(total_logits),
            training_epoch=self.last_epoch,
            data=self.data,
            tag=loader,
        )
        return closed_results

    def save(self, tag=None):
        filename = f"{self.output_dir}/{self.last_epoch}{f'.{tag}' if tag else ''}.pth"
        state_dict = self.model.state_dict()
        state_dict["cfg"] = self.model.network.cfg
        torch.save(state_dict, filename)
        self.log.info(f"Model saved as {filename}")
