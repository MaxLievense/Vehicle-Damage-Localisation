from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Union

import torch
from hydra.utils import instantiate

from src.utils.base import Base

warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

if TYPE_CHECKING:
    from src.data.data import Data
    from src.model.model import BaseModel


class BaseTrainer(Base):
    """
    Abstract Trainer class that handles the training loop and evaluation.
    All trainers should inherit from this class and implement the necessary methods.
    """

    def __init__(
        self,
        device: torch.device,
        model: BaseModel,
        data: Data,
        **cfg: dict,
    ):
        """
        Initializes the Trainer object.

        Args:
            device (torch.device): Device to run the training on.
            model (BaseModel): Model to be trained.
            data (Data): Data object.
            **cfg (dict): Configurations to be used.
        """
        super().__init__(cfg)
        self.device = device
        self.model = model
        self.data = data
        self.callback = instantiate(self.cfg.callbacks, _recursive_=False)

        self.optimizer, self.scheduler = self.init_optimizer()

        self.step = 0
        self.last_epoch = -1

        self.log.info("Loaded trainer.")

    def init_optimizer(self) -> tuple[torch.optim.Optimizer, Union[None, torch.optim.lr_scheduler._LRScheduler]]:
        """
        Initializes the optimizer and scheduler.

        Returns:
            tuple[torch.optim.Optimizer, Union[None, torch.optim.lr_scheduler._LRScheduler]]: The optimizer and
                scheduler objects, if configured.
        """
        optimizer = instantiate(self.cfg.optimizer, params=self.model.network.parameters())
        return (
            optimizer,
            (instantiate(self.cfg.scheduler, optimizer=optimizer) if self.cfg.get("scheduler") else None),
        )

    def run(self):
        """Main method to start the training process."""
        self.log.info("Running trainer...")
        self.train()
        self.log.info("Training finished.")
        if self.cfg.eval_at_end:
            self.eval()
        if self.cfg.save_last:
            self.save("end")

    def train(self):
        """
        Main method to run the training loop.

        Raises:
            NotImplementedError: This method must be implemented by any subclass.
        """
        raise NotImplementedError

    def eval(self, loader: str = "test"):
        """
        Main method to run the evaluation loop.

        Args:
            loader (str): The loader to evaluate on. Default: "test".
        Raises:
            NotImplementedError: This method must be implemented by any subclass.
        """
        raise NotImplementedError

    def inference(self, loader="test") -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inference loop for the Detection task.

        Args:
            loader (str): Loader to perform inference on. Defaults to "test".

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The outputs and targets.
        """
        self.model.network.eval()

        outputs = []
        targets = []
        with torch.no_grad():
            for _, data in enumerate(getattr(self.data, loader + "_loader")):
                data = [_data.to(self.device) if isinstance(_data, torch.Tensor) else _data for _data in data]
                _, output = self.model.step(self, *data)
                outputs.append(output)
                targets.append(data[1])
        return torch.cat(outputs), torch.cat(targets)

    def save(self, tag: str = None) -> None:
        """
        Saves the model state.

        Args:
            tag (str, optional): Tag to add to the filename. Defaults to None.

        """
        filename = f"{self.output_dir}/{self.last_epoch}{f'.{tag}' if tag else ''}.pth"
        state_dict = self.model.state_dict()
        torch.save(state_dict, filename)
        self.log.info(f"Model saved as {filename}")
