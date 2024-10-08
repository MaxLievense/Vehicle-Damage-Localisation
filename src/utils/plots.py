import logging

import matplotlib.pyplot as plt
from matplotlib import patches

logging.getLogger("matplotlib").setLevel(logging.ERROR)


def plot_img_with_bbox_and_gt(imgs, gt, pred):
    """
    Plots an image with bounding boxes and labels.

    Parameters:
    imgs (Tensor): Images tensor with shape (C, H, W).
    gt (list of dict): Ground truth bounding boxes and labels.
    pred (list of dict): Predicted bounding boxes, labels, and scores.
    """
    batch_size = imgs.size(0)
    fig, axs = plt.subplots(1, batch_size, figsize=(batch_size * 5, 5))
    for img, _gt, _pred, ax in zip(imgs, gt, pred, axs):
        img = img.permute(1, 2, 0).cpu().numpy()
        ax.imshow(img)
        gt_bboxes = _gt["boxes"].cpu().numpy()
        gt_labels = _gt["labels"].cpu().numpy()
        pred_bboxes = _pred["boxes"].cpu().numpy()[: len(gt_bboxes)]
        pred_labels = _pred["labels"].cpu().numpy()[: len(gt_bboxes)]
        pred_scores = _pred["scores"].cpu().numpy()[: len(gt_bboxes)]

        for bbox, label, score in zip(pred_bboxes, pred_labels, pred_scores):
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=2, edgecolor="r", facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(bbox[0], bbox[1], label, color="r", fontsize=12)
            ax.text(bbox[0], bbox[1] + 20, f"{score:.2f}", color="r", fontsize=12)

        for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
            rect = patches.Rectangle(
                (gt_bbox[0], gt_bbox[1]),
                gt_bbox[2] - gt_bbox[0],
                gt_bbox[3] - gt_bbox[1],
                linewidth=2,
                edgecolor="g",
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(gt_bbox[0], gt_bbox[1], gt_label, color="g", fontsize=12)
        ax.axis("off")
    plt.tight_layout()
    return fig
