import torch


class TensorDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to(self, *args, **kwargs):
        """Create a new TensorDict, where each tensor is moved to the desired device/dtype"""
        print([True if isinstance(v, torch.Tensor) else type(v) for k, v in self.items()])
        return TensorDict({k: v.to(*args, **kwargs) if isinstance(v, torch.Tensor) else v for k, v in self.items()})
