"""
Code adapted from: https://github.com/Tramac/awesome-semantic-segmentation-pytorch/blob/master/core/data/dataloader/mscoco.py
"""

import logging
import os

import torch
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
        self.ids = list(self.coco.anns.keys())
        self.transform = transform
        self.n_classes = len(self.coco.cats)

    def __getitem__(self, index):
        """Returns one data pair (image and bbox)."""
        coco = self.coco
        ann_id = self.ids[index]
        bbox = coco.anns[ann_id]["bbox"]
        img_id = coco.anns[ann_id]["image_id"]
        path = coco.loadImgs(img_id)[0]["file_name"]

        image = Image.open(os.path.join(self.root, path)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        target = torch.Tensor(bbox)
        return image, target

    def __len__(self):
        return len(self.coco.imgs)
