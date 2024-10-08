from __future__ import annotations

from copy import copy

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from ultralytics import YOLO

from src.data.data import Data
from src.trainer.trainer import Trainer


class YOLOTrainer(Trainer):
    # def __init__(
    #     self,
    #     device: torch.device,
    #     callbacks: dict,
    #     data: Data,
    #     **cfg: dict,
    # ) -> None:
    #     model = instantiate(cfg["model"])  # Instantiate YOLOv5 model
    #     _model_cfg = copy(cfg["model"])
    #     del cfg["model"]
    #     super().__init__(device, callbacks, model, data, **cfg)
    #     self.log.info(f"Loaded {_model_cfg['repo_or_dir']} model: {_model_cfg['model']}")

    def init_optimizer(self):
        self.optimizer = instantiate(DictConfig(self.cfg.optimizer), params=self.model.network.parameters())

    def train(self):
        """Training loop for YOLO."""
        self.log.info("Training YOLO model...")
        self.model.network.train()
        for epoch in range(self.cfg.epochs):
            epoch_loss = 0.0

            for data in self.data.train_loader:
                images, targets = images.to(self.device), targets.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()
                step_loss, logits = self.model.step(self, *data)
                self.optimizer.step()  # Optimize the model

                # Track epoch loss
                epoch_loss += loss.item()

    def eval(self, loader="test"):
        """Evaluation loop for YOLO."""
        self.log.info(f"Evaluating YOLO model on {loader} dataset...")
        _loader = getattr(self.data, loader + "_loader")

        # Track results
        total_labels = []
        total_logits = []
        self.model.eval()  # Set model to evaluation mode

        with torch.no_grad():
            for images, targets in _loader:
                images = images.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # Get model predictions
                outputs = self.model(images)
                preds = outputs["boxes"]  # Change according to YOLOv5 output format

                total_logits.append(preds)
                total_labels.append(targets)
