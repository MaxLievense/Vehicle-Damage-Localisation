"""
Dataset object to load a dataset in the COCO format.
"""

import logging
import os
from typing import Optional, Tuple, Union

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

logging.getLogger("PIL").setLevel(logging.WARNING)


class COCODataset(Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(
        self,
        root: str,
        json: str,
        transform: Optional[Union[torch.nn.Module, torch.nn.Sequential]] = None,
        min_area: Optional[float] = None,
    ):
        """
        Initialize COCO dataset.

        Args:
            root (str): Path to root directory to images.
            json (str): Path to annotation file.
            transform (torch.nn.Module, optional): Image transform pipeline, e.g., torchvision.transforms.Compose().
            min_area (float, optional): Filter the annotations by minimum area.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.n_classes = len(self.coco.cats) + 1  # Add 1 for background
        self.min_area = min_area

        if min_area is not None:
            self.ids = [
                img_id
                for img_id in self.ids
                if any(ann["area"] >= min_area for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id)))
            ]

        self.class_counts = {index: 0 for index in range(0, self.n_classes)}
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            if self.min_area is not None:
                anns = [ann for ann in anns if ann["area"] >= self.min_area]
            for ann in anns:
                self.class_counts[ann["category_id"]] += 1

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        """Returns the total number of images."""
        return len(self.ids)


class COCOMultibbox(COCODataset):
    """Exports COCO's multi-bounding box annotations."""

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Returns image data and corresponding bounding boxes.
        The bounding boxes are in the format (x_min, y_min, x_max, y_max).
        The target dictionary may contain multiple detections, where the index of is linked accross the keys.

        Note:
            Background images are annotated as zeros for each key retaining the same shape.

        Args:
            index (int): Index of the image.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                - image (torch.Tensor): The image tensor.
                - target (dict[str, torch.Tensor]): The target dictionary containing:
                    - boxes (torch.Tensor): The bounding boxes in the image.
                    - labels (torch.Tensor): The labels of the bounding boxes.
                    - image_id (torch.Tensor): The image ID.
                    - area (torch.Tensor): The area of the bounding boxes.
                    - iscrowd (torch.Tensor): The iscrowd flag for the bounding boxes.

        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        if self.min_area is not None:
            anns = [ann for ann in anns if ann["area"] >= self.min_area]

        path = coco.loadImgs(img_id)[0]["file_name"]
        image = Image.open(os.path.join(self.root, path)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if len(anns) > 0:
            target = {}
            boxes = []
            for box in [ann["bbox"] for ann in anns]:
                x, y, w, h = box
                boxes.append([x, y, x + w, y + h])
            target["boxes"] = torch.tensor(boxes)

            target["labels"] = torch.tensor([ann["category_id"] for ann in anns])
            target["image_id"] = torch.tensor([img_id])
            target["area"] = torch.tensor([ann["area"] for ann in anns])
            target["iscrowd"] = torch.tensor([ann["iscrowd"] for ann in anns])
        else:
            target = {}
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["image_id"] = torch.tensor([img_id])
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)
        return image, target
