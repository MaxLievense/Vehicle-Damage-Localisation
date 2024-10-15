from typing import Tuple

import torch


def collate_fn(batch: Tuple[torch.Tensor, dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, list]:
    """
    Custom collate function to handle different sized annotations in the same batch.

    Args:
        batch (Tuple[torch.Tensor, dict[str, torch.Tensor]]): Batch of data, image and target.
    Returns:
        Tuple[torch.Tensor, list]: Stacked images and list of targets, where all target values are the same size.
    """
    images = []
    targets = []

    for item in batch:
        image, target = item
        images.append(image)
        targets.append(target)
    images = torch.stack(images, dim=0)
    return images, targets
