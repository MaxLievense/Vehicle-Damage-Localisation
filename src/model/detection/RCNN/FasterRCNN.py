from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torchvision

from src.model.model import BaseModel

if TYPE_CHECKING:
    from src.data.data import Data
    from src.trainer.trainer import BaseTrainer as Trainer

torchvision.models.detection.fasterrcnn_resnet50_fpn_v2


class FasterRCNN(BaseModel):
    """
    Interface for Faster R-CNN model.
    """

    def init_network(self) -> torch.nn.Module:
        """
        Fixes the number of classes in the FastRCNNPredictor.

        Returns:
            torch.nn.Module: The Faster R-CNN model.
        """
        network = super().init_network()
        in_features = network.roi_heads.box_predictor.cls_score.in_features
        network.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, self.n_classes + 1
        ).to(self.device)
        return network

    def step(
        self, trainer: Trainer, data: Data, target: dict[str, torch.Tensor]
    ) -> tuple[float, dict[str, torch.Tensor]]:
        """
        Custom step function for Faster R-CNN.

        Args:
            trainer (Trainer): Trainer object.
            data (Data): Data object.
            target (dict[str, torch.Tensor]): Target dictionary.

        Returns:
            tuple[float, dict[str, torch.Tensor]]: Loss and output dictionary.
        """
        if self.network.training:
            with torch.set_grad_enabled(True):
                # TODO: The Dataloader returns tensors on CPU, trainer only sends to device if tensor, not dict.
                for i, _target in enumerate(target):
                    for k, v in _target.items():
                        target[i][k] = v.to(self.device)
                output = self.network.forward(data, target)
                loss = sum(list(output.values()))
                loss.backward()

                if self.cfg.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.cfg.clip_grad)
                return loss.item(), output
        output = self.network.forward(data)
        return None, output
