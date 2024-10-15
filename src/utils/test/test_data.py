from typing import List, Tuple

import pytest
import torch

from src.utils.data import collate_fn


def test_collate_fn():
    """Tests if the collate_fn unifies the lengths of targets in the batch."""
    image1 = torch.randn(3, 224, 224)
    image2 = torch.randn(3, 224, 224)
    target1 = {"boxes": torch.tensor([[0, 0, 10, 10]]), "labels": torch.tensor([1])}
    target2 = {"boxes": torch.tensor([[20, 20, 50, 50], [0, 0, 10, 10]]), "labels": torch.tensor([2, 3])}
    batch = [(image1, target1), (image2, target2)]

    images, targets = collate_fn(batch)

    assert isinstance(images, torch.Tensor), "Images output should be a torch.Tensor"
    assert isinstance(targets, list), "Targets output should be a list"
    assert images.shape == (2, 3, 224, 224), f"Expected image tensor shape (2, 3, 224, 224), but got {images.shape}"

    assert len(targets) == 2, "There should be 2 target dictionaries in the output"
    assert isinstance(targets[0], dict), "Each target should be a dictionary"
    assert (
        "boxes" in targets[0] and "labels" in targets[0]
    ), "Each target dictionary should have 'boxes' and 'labels' keys"
    assert torch.equal(targets[0]["boxes"], target1["boxes"]), "Boxes in target 1 do not match"
    assert torch.equal(targets[1]["boxes"], target2["boxes"]), "Boxes in target 2 do not match"
