from dataclasses import dataclass

from torchvision.ops import box_iou


@dataclass
class DetectionMetrics:
    """Dataclass to store detection metrics."""

    Precision: float = 0
    Recall: float = 0
    F1_Score: float = 0
    mAP: float = 0
    UOI: float = 0
    count: int = 0

    def __str__(self):
        """Returns a string representation of the DetectionMetrics object."""
        return (
            f"Precision: {self.Precision:.4f}, Recall: {self.Recall:.4f}, F1 Score: {self.F1_Score:.4f}, "
            + f"mAP: {self.mAP:.4f}, UOI: {self.UOI:.4f}"
        )

    def append(self, other):
        """
        Appends another DetectionMetrics object to the current object.
        Ensurses the weight of both objects is kept, based on the `self.count`.
        """
        if other.count > 0:
            self.Precision = (self.Precision * self.count + other.Precision * other.count) / (self.count + other.count)
            self.Recall = (self.Recall * self.count + other.Recall * other.count) / (self.count + other.count)
            self.F1_Score = (self.F1_Score * self.count + other.F1_Score * other.count) / (self.count + other.count)
            self.mAP = (self.mAP * self.count + other.mAP * other.count) / (self.count + other.count)
            self.UOI = (self.UOI * self.count + other.UOI * other.count) / (self.count + other.count)
            self.count += other.count
        return self

    def to_wandb(self, tag):
        """Converts the DetectionMetrics object to a dictionary for logging to Weights & Biases."""
        return {
            f"{tag}/Precision": self.Precision,
            f"{tag}/Recall": self.Recall,
            f"{tag}/F1 Score": self.F1_Score,
            f"{tag}/mAP": self.mAP,
            f"{tag}/UOI": self.UOI,
        }


def evaluate_detection_metrics(ground_truth, predictions, results: DetectionMetrics, iou_threshold=0.5):
    """
    Evaluates object detection metrics including True Positives, False Positives, False Negatives,
    Precision, Recall, F1 Score, and mean Average Precision (mAP).

    Parameters:
    ground_truth (list of dict): List of ground truth boxes, labels, and other information.
    predictions (list of dict): List of predicted boxes, labels, and confidence scores.
    iou_threshold (float): IoU threshold to determine true positives.

    Returns:
    dict: A dictionary containing evaluation metrics: TP, FP, FN, Precision, Recall, F1 Score, mAP, and UOI.
    """
    tp, fp, fn = 0, 0, 0
    aps = []
    iou_values = []

    for gt, pred in zip(ground_truth, predictions):
        if gt["boxes"].size(0) == 0 or pred["boxes"].size(0) == 0:
            fn += len(gt["boxes"])
            fp += len(pred["boxes"])
            aps.append(0)  # Append 0 AP if no predictions for this image
            continue

        ious = box_iou(gt["boxes"].cpu(), pred["boxes"].cpu())

        matched_gt = set()
        for i, row in enumerate(ious):
            for j, iou in enumerate(row):
                if iou > iou_threshold and gt["labels"][i] == pred["labels"][j] and i not in matched_gt:
                    tp += 1
                    matched_gt.add(i)
                    iou_values.append(iou.item())
                    break

        fp += len(pred["boxes"]) - len(matched_gt)
        fn += len(gt["boxes"]) - len(matched_gt)

        max_ious, _ = ious.max(0)
        img_tp = (max_ious >= iou_threshold).sum().item()
        img_fp = (max_ious < iou_threshold).sum().item()

        precision = img_tp / (img_tp + img_fp) if (img_tp + img_fp) > 0 else 0
        aps.append(precision)  # Precision used for mAP calculation

    mean_ap = sum(aps) / len(aps) if aps else 0
    precision, recall, f1_score = calculate_precision_recall_f1(tp, fp, fn)
    uoi = sum(iou_values) / len(iou_values) if iou_values else 0

    results.append(DetectionMetrics(precision, recall, f1_score, mean_ap, uoi, count=len(ground_truth)))


def calculate_precision_recall_f1(tp, fp, fn):
    """Helper function to calculate precision, recall, and F1 score."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score
