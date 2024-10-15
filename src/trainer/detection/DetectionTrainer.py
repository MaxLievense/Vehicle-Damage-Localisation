from __future__ import annotations

import warnings

import torch
import wandb

from src.trainer.trainer import BaseTrainer
from src.utils.metrics import DetectionMetrics, evaluate_detection_metrics
from src.utils.plots import plot_img_with_bbox_and_gt

warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")


class DetectionTrainer(BaseTrainer):
    """Trainer for the Detection task."""

    def train(self):
        """Detection training loop."""
        self.model.network.train()
        self.model.pretrain(self)

        self.log.info("Training...")
        for epoch in range(self.cfg.epochs):
            self.last_epoch = epoch
            _breaking = False
            self.model.preepoch(self)

            running_loss = 0.0
            training_entries = 0
            for data in self.data.train_loader:
                training_entries += len(data[0])
                self.model.network.train()
                data = [_data.to(self.device) if isinstance(_data, torch.Tensor) else _data for _data in data]
                self.optimizer.zero_grad()
                step_loss, logits = self.model.step(self, *data)
                self.optimizer.step()
                running_loss += step_loss
            _lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]["lr"]
            self.log.info(
                f"Epoch {epoch+1}/{self.cfg.epochs} | "
                + f"avg loss: {running_loss / training_entries:.6f} "
                + f"(lr: {_lr:.5f})"
            )

            if self.callback(
                trainer=self,
                loss=running_loss,
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
        """
        Evaluation loop for the Detection task.

        Args:
            loader (str): Loader to evaluate on. Defaults to "test".
        """

        if loader == "test":
            self.log.info("Evaluating...")

        self.model.network.eval()

        running_results = DetectionMetrics()
        with torch.no_grad():
            for _, data in enumerate(getattr(self.data, loader + "_loader")):
                data = [_data.to(self.device) if isinstance(_data, torch.Tensor) else _data for _data in data]
                _, output = self.model.step(self, *data)
                evaluate_detection_metrics(output, data[1], running_results)
        wandb.log(running_results.to_wandb(loader), commit=False)
        self.log.info(f"Loader {loader}: " + str(running_results))

        if self.cfg.preview:
            fig = plot_img_with_bbox_and_gt(data[0], data[1], output)
            wandb.log({f"{loader}/images": [wandb.Image(fig)]}, commit=False)
