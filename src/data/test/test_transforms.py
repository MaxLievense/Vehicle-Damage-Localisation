from torchvision import transforms

from src.data.data import Data
from src.utils.test import MockSelf


def test_get_transform():
    """Test for the get_transform method."""

    mock_self = MockSelf(
        {
            "dataset": {
                "dataset_transforms": [
                    {"torchvision.transforms.RandomHorizontalFlip": None},
                    {"torchvision.transforms.RandomCrop": {"size": 32}},
                    {"torchvision.transforms.ToTensor": None},
                ],
                "training_transforms": [
                    {"torchvision.transforms.ColorJitter": {"brightness": 0.5}},
                    {"torchvision.transforms.ToTensor": None},
                ],
            }
        }
    )

    plain_transform, training_transform = Data.get_transform(mock_self)

    assert all(
        [transform in training_transform.transforms for transform in plain_transform.transforms]
    ), "Plain transforms should be a subset of training transforms"
    assert (
        len([transform for transform in training_transform.transforms if isinstance(transform, transforms.ToTensor)])
        == 1
    ), "Duplicate ToTensor transform should have been removed"
