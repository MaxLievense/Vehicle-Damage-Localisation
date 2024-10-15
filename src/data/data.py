from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Tuple

import torch
from hydra.utils import instantiate
from torchvision import transforms

from src.utils.base import Base
from utils.imports import import_from_cfg

if TYPE_CHECKING:
    from torch.utils.data import DataLoader, Dataset


class Data(Base):
    """
    Data module to handle dataset loading and data loaders.

    This module should be as generic as possible and should be able to handle any dataset, where the specific dataset
        handling is done in the dataset submodules.
    """

    def __init__(
        self,
        device: torch.device,
        dataset_transforms: Optional[list] = [],
        training_transforms: Optional[list] = [],
        **cfg: dict,
    ):
        """
        Initialize the Data class.

        Args:
            device (torch.device): Device to run the model on.
            dataset_transforms (list, optional): List of dataset transforms. Defaults to [].
            model_transforms (list, optional): List of model transforms. Defaults to [].
            training_transforms (list, optional): List of training transforms. Defaults to [].
            **cfg (dict): Remaining Hydra configurations.
        """
        super().__init__(cfg)

        self.dataset_name = self.cfg.dataset.dataset_name
        self.device = device

        self.plain_transform, self.training_transform = self.get_transform(dataset_transforms, training_transforms)
        self.log.debug(f"Plain transforms: {self.plain_transform}")
        self.log.debug(f"Training transforms: {self.training_transform}")

        self.train_data, self.train_plain_data, self.test_data, self.val_data = self.get_dataset()
        self.train_loader, self.train_plain_loader, self.val_loader, self.test_loader = self.get_loaders()
        self.n_classes = self.test_data.n_classes if hasattr(self.test_data, "n_classes") else None
        self.log.info(
            f"Loaded {self.dataset_name} dataset with {self.n_classes} classes, "
            + f"{len(self.train_data)} train-, {len(self.val_data)} val-, and {len(self.test_data)} test datapoints."
        )

    def get_transform(
        self, pre_dataset_transforms: list, pre_training_transforms: list
    ) -> Tuple[transforms.Compose, transforms.Compose]:
        """
        Generates the transforms based on the configuration.
        Args:
            pre_dataset_transforms (list): List of dataset transforms.
            pre_training_transforms (list): List of training transforms.
        Returns:
            Tuple[transforms.Compose, transforms.Compose]: Plain and training transforms.
        """

        def build_transforms(transform):
            transform_class, transform_attr = list(transform.items())[0]
            transform_attr = {} if transform_attr is None else transform_attr
            if not transform_class.startswith("torchvision"):
                return import_from_cfg(transform_class)(**transform_attr).get()
            return import_from_cfg(transform_class)(**transform_attr)

        dataset_transforms = [build_transforms(transform) for transform in pre_dataset_transforms]
        training_transforms = [build_transforms(transform) for transform in pre_training_transforms]
        return transforms.Compose(dataset_transforms), transforms.Compose(training_transforms + dataset_transforms)

    def get_dataset(self) -> Tuple[Dataset, Dataset, Dataset, Dataset]:
        """
        Gets the dataset based on the configuration.
        Creates a plain train dataset loader which does not apply any training transforms.
        Imports the collate function if specified in the configuration.

        Returns:
            Tuple[Dataset, Dataset, Dataset]: Train, test, and validation datasets.
        """
        train_data = instantiate(self.cfg.dataset.train, transform=self.training_transform)
        train_plain_data = deepcopy(train_data)
        train_plain_data.transform = self.plain_transform
        self._collate_fn = import_from_cfg(self.cfg.dataset.collate_fn) if self.cfg.dataset.collate_fn else None

        return (
            train_data,
            train_plain_data,
            instantiate(self.cfg.dataset.test, transform=self.plain_transform),
            instantiate(self.cfg.dataset.val, transform=self.plain_transform),
        )

    def get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        """
        Creates and returns data loaders for training, validation, and testing datasets.
        Adds collate_fn to the data loaders if specified in the configuration.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader, DataLoader]: A tuple containing the train loader,
                plain train loader, validation loader, and test loader.
        """

        def _get_loader(loader_cfg, dataset):
            """
            Helper function to instantiate a data loader based on the given configuration and dataset.

            Args:
                loader_cfg (dict): Configuration dictionary for the data loader.
                dataset (Dataset): The dataset for which the loader will be created.

            Returns:
                DataLoader: Instantiated data loader.
            """
            sampler = instantiate(loader_cfg.sampler, dataset=dataset) if loader_cfg.get("sampler") else None

            if sampler and loader_cfg.shuffle:
                self.log.warning("Sampler and shuffle are both set to True. Sampler will be used.")
                loader_cfg.shuffle = False
            return instantiate(loader_cfg, dataset=dataset, sampler=sampler, collate_fn=self._collate_fn)

        return (
            _get_loader(self.cfg.dataloaders.train_loader, dataset=self.train_data),
            _get_loader(self.cfg.dataloaders.train_plain_loader, dataset=self.train_plain_data),
            _get_loader(self.cfg.dataloaders.val_loader, dataset=self.val_data),
            _get_loader(self.cfg.dataloaders.test_loader, dataset=self.test_data),
        )
