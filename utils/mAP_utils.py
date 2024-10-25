############## Below is the code for mAP calculation #####################
import numpy as np

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Coordinates of intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap

    # Intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Union area
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area
    return iou

def calculate_ap(precision, recall):
    """
    Calculate Average Precision (AP) from precision and recall values.
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate area under curve
    indices = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = np.sum((mrec[indices] - mrec[indices - 1]) * mpre[indices])
    return ap

def calculate_map(predictions, labels, num_classes=2, iou_threshold =0.5):
    """
    input
        predictions format: ndarray, [class_id, score, x, y, w, h]
        label format: ndarray, [class_id, x, y, w, h]
    Calculate mean Average Precision (mAP) for object detection.
    """
    # Sort predictions by confidence score
    predictions = predictions[predictions[:, 1].argsort()][::-1]

    average_precisions = []
    for class_id in range(num_classes):  # Assuming 2 classes
        true_positives = np.zeros(len(predictions))
        false_positives = np.zeros(len(predictions))
        num_gt_boxes = 0

        for i, pred in enumerate(predictions):
            # pred_class, confidence, x, y, w, h = pred
            pred_class, x, y, w, h = pred
            if pred_class != class_id:
                continue

            pred_box = [x, y, w, h]

            # Find matching ground truth box
            best_iou = 0

            for label in labels:
                if label[0] != class_id:
                    continue
                # iou = calculate_iou(pred_box, label[1:])
                iou = computeIoU(pred_box, label[1:])
                if iou > best_iou:
                    best_iou = iou
            if best_iou >= iou_threshold:
                true_positives[i] = 1
            else:
                false_positives[i] = 1

        # Handle case of no ground truth boxes for this class
        if np.sum(true_positives) == 0:
            average_precisions.append(0)
            continue

        # Compute precision and recall
        cumsum = np.cumsum(true_positives)
        precision = cumsum / (np.arange(len(predictions)) + 1)
        recall = cumsum / np.sum(true_positives)

        # Calculate Average Precision (AP) for this class
        ap = calculate_ap(precision, recall)
        average_precisions.append(ap)

    # Calculate mAP
    mAP = np.nanmean(average_precisions)
    # mAP = np.nanmean(average_precisions[0])
    return mAP


def computeIoU(bbox1, bbox2):
    (x1, y1, w1, h1) = bbox1
    (x2, y2, w2, h2) = bbox2

    # Firstly, we calculate the areas of each box
    # by multiplying its height with its width.
    area1 = w1 * h1
    area2 = w2 * h2

    # Secondly, we determine the intersection
    # rectangle. For that, we try to find the
    # corner points (top-left and bottom-right)
    # of the intersection rectangle.
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    # From the two corner points we compute the
    # width and height.
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    # If the width or height are equal or less than zero
    # the boxes do not overlap. Hence, the IoU equals 0.
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    # Otherwise, return the IoU (intersection area divided
    # by the union)
    else:
        inter_area = inter_w * inter_h
        return inter_area / float(area1 + area2 - inter_area)
