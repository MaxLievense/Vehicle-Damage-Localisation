"""
Code adapted from: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/data_loader.py
"""

import logging
import os

import torch
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

logging.getLogger("PIL").setLevel(logging.WARNING)


class COCODataset(Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, transform=None):
        """Set the path for images and captions.

        Args:
            root: image directory.
            json: coco annotation file path.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.n_classes = len(self.coco.cats)

    def __getitem__(self, index):
        """Returns one data pair (image and bbox)."""
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]["file_name"]

        image = Image.open(os.path.join(self.root, path)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.ids)
