from __future__ import annotations

import warnings

import torch
import wandb

from src.trainer.ClassificationTrainer import ClassificationTrainer
from src.utils.metrics import evaluate_detection_metrics

warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")


class DetectionTrainer(ClassificationTrainer):
    def train(self):
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
                accuracy=None,
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

        self.model.network.eval()

        running_results = None
        eval_entries = 0
        with torch.no_grad():
            for _, data in enumerate(getattr(self.data, loader + "_loader")):
                eval_entries += len(data[0])
                data = [_data.to(self.device) if isinstance(_data, torch.Tensor) else _data for _data in data]
                _, output = self.model.step(self, *data)
                results = evaluate_detection_metrics(output, data[1])
                if running_results is None:
                    running_results = results
                else:
                    for key in running_results.keys():
                        running_results[key] += results[key]
        final_results = {f"{loader}/{key}": value / eval_entries for key, value in running_results.items()}
        wandb.log(final_results)
