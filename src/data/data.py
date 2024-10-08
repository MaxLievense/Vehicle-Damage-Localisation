from copy import deepcopy

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from src.utils.base import Base
from src.utils.module import import_from_cfg


class Data(Base):
    def __init__(
        self,
        device: torch.device,
        dataset_transforms: list,
        model_transforms: list,
        training_transforms: list,
        **cfg: dict,
    ) -> None:
        super().__init__(cfg)
        self.log.debug(f"Building dataset: {self.cfg.dataset.dataset_name}")

        self.dataset_name = self.cfg.dataset.dataset_name
        self.device = device

        self.get_transform(model_transforms, dataset_transforms, training_transforms)
        self.get_dataset(DictConfig(self.cfg.dataset))
        self.get_loaders(DictConfig(self.cfg.dataloaders))
        self.n_classes = self.test_data.n_classes if hasattr(self.test_data, "n_classes") else None

        self.log.info(
            f"Loaded {self.dataset_name} dataset with {self.n_classes} classes, "
            + f"{len(self.train_data)} train-, {len(self.val_data)} val-, and {len(self.test_data)} test datapoints."
        )

    def get_transform(
        self, pre_model_transforms: list = [], pre_dataset_transforms: list = [], pre_training_transforms: list = []
    ) -> None:
        def build_transforms(transform):
            transform_class, transform_attr = list(transform.items())[0]
            transform_attr = {} if transform_attr is None else transform_attr
            if not transform_class.startswith("torchvision"):
                return import_from_cfg(transform_class)(**transform_attr).get()
            return import_from_cfg(transform_class)(**transform_attr)

        self.log.debug("Building transforms...")

        post_model_transforms = [build_transforms(transform) for transform in pre_model_transforms]
        post_dataset_transforms = [build_transforms(transform) for transform in pre_dataset_transforms]
        post_training_transforms = [build_transforms(transform) for transform in pre_training_transforms]

        self.plain_transform = transforms.Compose(post_model_transforms + post_dataset_transforms)
        self.log.debug(f"Plain transforms: {self.plain_transform}")

        RESIZE_TRANSFORMS = (transforms.Resize, transforms.RandomResizedCrop, transforms.CenterCrop)  # Order matters.
        if any(isinstance(transform, RESIZE_TRANSFORMS) for transform in post_training_transforms) and any(
            _model_resize_idx := [isinstance(transform, RESIZE_TRANSFORMS) for transform in post_model_transforms]
        ):
            _model_resize_idx.reverse()
            input_size = post_model_transforms[_model_resize_idx.index(True)].size
            for transform in post_training_transforms:
                if isinstance(transform, RESIZE_TRANSFORMS):
                    transform.size = input_size

        self.training_transform = transforms.Compose(post_training_transforms + post_dataset_transforms)
        self.log.debug(f"Training transforms: {self.training_transform}")

    def get_dataset(self, dataset_cfg: DictConfig) -> None:
        self.log.debug("Building dataset...")

        self._init_dataset(dataset_cfg)
        # self._limit_dataset()

        self.train_plain_data = deepcopy(self.train_data)
        self.train_plain_data.transform = self.plain_transform
        assert len(self.train_data) == len(self.train_plain_data)

    def _init_dataset(self, dataset_cfg: DictConfig):
        self.train_data, self.test_data, self.val_data = (
            instantiate(dataset_cfg.train, transform=self.training_transform),
            instantiate(dataset_cfg.test, transform=self.plain_transform),
            instantiate(dataset_cfg.val, transform=self.plain_transform),
        )

    def _limit_dataset(self):
        raise NotImplementedError
        if self.cfg.limit.train and self.train_data and self.cfg.limit.train < len(self.train_data):
            self.train_data, _ = split_dataset(self.train_data, self.cfg.limit.train, self.cfg.seed)
            self.log.info(f"\tLimited train data to {len(self.train_data)} samples")
        if self.cfg.limit.val and self.val_data and self.cfg.limit.val < len(self.val_data):
            self.val_data, _ = split_dataset(self.val_data, self.cfg.limit.val, self.cfg.seed)
            self.log.info(f"\tLimited val data to {len(self.val_data)} samples")
        if self.cfg.limit.test and self.test_data and self.cfg.limit.test < len(self.test_data):
            self.test_data, _ = split_dataset(self.test_data, self.cfg.limit.test, self.cfg.seed)
            self.log.info(f"\tLimited test data to {len(self.test_data)} samples")

    def get_loaders(self, dataloader_cfg: DictConfig) -> None:
        def _get_loader(loader_cfg, dataset):
            sampler = instantiate(loader_cfg.sampler, dataset=dataset) if loader_cfg.get("sampler") else None
            if sampler and loader_cfg.shuffle:
                self.log.warning("Sampler and shuffle are both set to True. Sampler will be used.")
                loader_cfg.shuffle = False
            return instantiate(loader_cfg, dataset=dataset, sampler=sampler, collate_fn=self._collate_fn)

        self._collate_fn = getattr(self, self.cfg.collate_fn) if self.cfg.collate_fn else None

        self.train_loader = _get_loader(dataloader_cfg.train_loader, dataset=self.train_data)
        self.train_plain_loader = _get_loader(dataloader_cfg.train_plain_loader, dataset=self.train_plain_data)
        self.val_loader = _get_loader(dataloader_cfg.val_loader, dataset=self.val_data)
        self.test_loader = _get_loader(dataloader_cfg.test_loader, dataset=self.test_data)

    @staticmethod
    def collate_fn(batch):
        images = []
        targets = []

        for item in batch:
            # Separate the image tensor and the dictionary containing metadata
            image, target = item

            # Append the image tensor and target dictionary separately
            images.append(image)
            targets.append(target)

        # Stack the image tensors into a single batch tensor
        images = torch.stack(images, dim=0)

        # The targets should remain a list of dictionaries
        return images, targets
