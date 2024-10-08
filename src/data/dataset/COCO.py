import logging
import os

import torch
import wandb
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

logging.getLogger("PIL").setLevel(logging.WARNING)


class COCODataset(Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, transform=None, min_area=None):
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
        self.min_area = min_area

        if min_area is not None:
            self.ids = [
                img_id
                for img_id in self.ids
                if any(ann["area"] >= min_area for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id)))
            ]

        self.category_id_counts = {index: 0 for index in range(0, self.n_classes)}
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            if self.min_area is not None:
                anns = [ann for ann in anns if ann["area"] >= self.min_area]
            for ann in anns:
                self.category_id_counts[ann["category_id"]] += 1
        print(self.category_id_counts)

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.ids)


# class COCOSinglebbox(COCODataset):
#     """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

#     def __init__(self, root, json, transform=None):
#         """Set the path for images and captions.

#         Args:
#             root: image directory.
#             json: coco annotation file path.
#             transform: image transformer.
#         """
#         self.root = root
#         self.coco = COCO(json)
#         self.ids = list(self.coco.anns.keys())
#         self.transform = transform
#         self.n_classes = len(self.coco.cats)

#     def __getitem__(self, index):
#         """Returns one data pair (image and bbox)."""
#         ann_id = self.ids[index]
#         bbox = self.coco.anns[ann_id]["bbox"]
#         img_id = self.coco.anns[ann_id]["image_id"]
#         path = self.coco.loadImgs(img_id)[0]["file_name"]

#         image = Image.open(os.path.join(self.root, path)).convert("RGB")
#         if self.transform is not None:
#             image = self.transform(image)

#         target = torch.Tensor(bbox)
#         return image, target


class COCOMultibbox(COCODataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __getitem__(self, index):
        """Returns all data from image (image and bboxes)."""
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
