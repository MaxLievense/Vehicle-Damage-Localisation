import pytest
import torch

from src.utils.metrics import DetectionMetrics, evaluate_detection_metrics


def test_detection_metrics():
    results = DetectionMetrics()
    new_results = DetectionMetrics(Precision=0.5, Recall=0.5, F1_Score=0.5, mAP=0.5, UOI=0.5, count=1)
    results.append(new_results)
    assert results.Precision == results.Recall == results.F1_Score == results.mAP == results.UOI == 0.5
    assert results.count == 1

    new_results = DetectionMetrics(Precision=1, Recall=0, F1_Score=0, mAP=0, UOI=0, count=3)
    results.append(new_results)
    assert results.Precision == 0.875
    assert results.Recall == results.F1_Score == results.mAP == results.UOI == 0.125
    assert results.count == 4


@pytest.fixture
def setup_data():
    ground_truth = [
        {"boxes": torch.tensor([[10, 10, 20, 20]]), "labels": torch.tensor([1])},
        {"boxes": torch.tensor([[30, 30, 40, 40]]), "labels": torch.tensor([2])},
        {"boxes": torch.tensor([]), "labels": torch.tensor([])},  # Empty ground truth for one case
    ]

    predictions = [
        {"boxes": torch.tensor([[12, 12, 18, 18]]), "labels": torch.tensor([1]), "scores": torch.tensor([0.9])},
        {"boxes": torch.tensor([[28, 28, 41, 41]]), "labels": torch.tensor([2]), "scores": torch.tensor([0.7])},
        {"boxes": torch.tensor([[50, 50, 60, 60]]), "labels": torch.tensor([3]), "scores": torch.tensor([0.8])},
    ]

    return ground_truth, predictions


def test_evaluate_detection_metrics_empty_input():
    """Test case with empty input"""
    results = DetectionMetrics()
    evaluate_detection_metrics([], [], results)

    assert results.Precision == 0
    assert results.Recall == 0
    assert results.F1_Score == 0
    assert results.mAP == 0
    assert results.UOI == 0
    assert results.count == 0


def test_evaluate_detection_metrics_iou_threshold(setup_data):
    """Test case with different IoU thresholds"""
    ground_truth, predictions = setup_data
    results = DetectionMetrics()
    evaluate_detection_metrics(ground_truth, predictions, results, iou_threshold=0.3)

    # Validating the results
    assert results.count == len(ground_truth), "Count of images evaluated should match ground truth length"
    assert results.Precision > 0, "Precision should have a positive value"
    assert results.Recall > 0, "Recall should have a positive value"
    assert results.F1_Score > 0, "F1 Score should have a positive value"
    assert results.mAP > 0, "Mean Average Precision should have a positive value"

    new_results = DetectionMetrics()
    evaluate_detection_metrics(ground_truth, predictions, new_results, iou_threshold=0.5)

    assert new_results.Precision < results.Precision, "Precision should decrease with higher IoU threshold"
    assert new_results.Recall < results.Recall, "Recall should decrease with higher IoU threshold"
    assert new_results.mAP < results.mAP, "Mean Average Precision should be affected by IoU threshold"

    assert new_results.UOI > results.UOI, "UOI should increase with higher IoU threshold"


def test_evaluate_detection_metrics_mismatched_labels():
    """Case with mismatched labels between ground truth and predictions"""
    ground_truth = [{"boxes": torch.tensor([[10, 10, 20, 20]]), "labels": torch.tensor([1])}]
    predictions = [
        {"boxes": torch.tensor([[12, 12, 18, 18]]), "labels": torch.tensor([2]), "scores": torch.tensor([0.9])}
    ]

    results = DetectionMetrics()
    evaluate_detection_metrics(ground_truth, predictions, results, iou_threshold=0.5)
    assert results.UOI == 0, "UOI should be zero when labels are mismatched"
    assert results.mAP == 0, "Mean AP should be zero when no correct predictions are made"
