from __future__ import annotations

import warnings

import torch

from src.trainer.trainer import BaseTrainer

warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")


class ClassificationTrainer(BaseTrainer):
    """Trainer for the Classification task."""

    def train(self):
        """Classification training loop."""
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
