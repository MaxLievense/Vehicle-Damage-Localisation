import torch

from src.utils.base import Base


class TorchHub(torch.nn.Module, Base):
    def __init__(self, n_classes: int, device: torch.device, **cfg):
        super().__init__()
        Base.__init__(self, cfg)
        self.network = torch.hub.load(self.cfg.url, self.cfg.tag, pretrained=self.cfg.pretrained)

    def forward(self, x):
        return self.network(x)
