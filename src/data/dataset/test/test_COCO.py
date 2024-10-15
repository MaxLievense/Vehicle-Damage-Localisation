import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from src.data.dataset.COCO import COCODataset, COCOMultibbox


def create_dummy_image(path, size=(224, 224)):
    image = Image.new("RGB", size)
    image.save(path)


@pytest.fixture(scope="module")
def setup_dummy_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        json_data = {
            "images": [{"id": 1, "file_name": "image1.jpg"}, {"id": 2, "file_name": "image2.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 200, 200], "area": 40000, "iscrowd": 0},
                {"id": 2, "image_id": 2, "category_id": 1, "bbox": [50, 50, 5, 5], "area": 25, "iscrowd": 0},
            ],
            "categories": [{"id": 1, "name": "person"}],
        }
        json_path = os.path.join(tmpdir, "dummy_coco.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f)

        create_dummy_image(os.path.join(tmpdir, "image1.jpg"))
        create_dummy_image(os.path.join(tmpdir, "image2.jpg"))
        yield tmpdir, json_path


def test_coco_dataset_len(setup_dummy_data):
    """Test the length of COCODataset."""
    tmpdir, json_path = setup_dummy_data
    dataset = COCODataset(root=tmpdir, json=json_path)
    assert len(dataset) == 2


def test_coco_multibbox_getitem(setup_dummy_data):
    """Test the __getitem__ method of COCOMultibbox."""
    tmpdir, json_path = setup_dummy_data
    dataset = COCOMultibbox(root=tmpdir, json=json_path)

    transform_mock = MagicMock()
    dataset.transform = transform_mock

    image, target = dataset[0]
    assert isinstance(image, torch.Tensor) or transform_mock.called
    assert isinstance(target, dict)
    assert "boxes" in target
    assert "labels" in target
    assert "area" in target
    assert "iscrowd" in target

    assert target["boxes"].shape == (1, 4)
    assert target["labels"].shape == (1,)
    assert target["area"].shape == (1,)
    assert target["iscrowd"].shape == (1,)

    image, target = dataset[1]
    assert isinstance(image, torch.Tensor) or transform_mock.called
    assert target["boxes"].shape == (1, 4)
    assert target["labels"].shape == (1,)
    assert target["area"].shape == (1,)


def test_coco_dataset_min_area_filter(setup_dummy_data):
    """Test the min_area filtering feature."""
    tmpdir, json_path = setup_dummy_data
    dataset = COCODataset(root=tmpdir, json=json_path, min_area=500)
    assert len(dataset) == 1
