import torch
import torchvision

from src.model.model import BaseModel


class FasterRCNN(BaseModel):
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
                loss = sum(list(output.values()))
                loss.backward()

                if self.cfg.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.cfg.clip_grad)
                return loss.item(), output
        output = self.network(data)
        return None, output
