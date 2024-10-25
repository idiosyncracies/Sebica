

def get_quadrant(x_center, y_center):
    # 确定框在哪个象限：左上、右上、左下、右下（此时x_center、y_center已经是归一化的）
    if x_center < 0.5 and y_center < 0.5:
        return "top_left"
    elif x_center >= 0.5 and y_center < 0.5:
        return "top_right"
    elif x_center < 0.5 and y_center >= 0.5:
        return "bottom_left"
    else:
        return "bottom_right"

def merge_boxes(box1, box2):
    # 解包框的参数
    class1, confidence1, x1_center, y1_center, w1, h1 = box1
    class2, confidence2, x2_center, y2_center, w2, h2 = box2

    # 计算框的面积
    area1 = w1 * h1
    area2 = w2 * h2

    # 计算合并后的框（已归一化）
    x_min = min(x1_center - w1 / 2, x2_center - w2 / 2)
    x_max = max(x1_center + w1 / 2, x2_center + w2 / 2)
    y_min = min(y1_center - h1 / 2, y2_center - h2 / 2)
    y_max = max(y1_center + h1 / 2, y2_center + h2 / 2)

    # 重新计算中心点和宽高
    new_x_center = (x_min + x_max) / 2
    new_y_center = (y_min + y_max) / 2
    new_w = x_max - x_min
    new_h = y_max - y_min

    # 选择面积大的 class 和 confidence
    if area1 >= area2:
        new_class = class1
        new_confidence = confidence1
    else:
        new_class = class2
        new_confidence = confidence2

    return [new_class, new_confidence, new_x_center, new_y_center, new_w, new_h]


def merge_boxes_near_boundary(boxes, threshold_ratio):
    threshold = threshold_ratio  # 阈值已归一化，直接用
    merged_boxes = []
    skip_indices = set()  # 用来存放已合并的框

    for i in range(len(boxes)):
        if i in skip_indices:
            continue
        box1 = boxes[i]
        class1, confidence1, x1_center, y1_center, w1, h1 = box1[:6]
        quadrant1 = get_quadrant(x1_center, y1_center)

        merged = False
        for j in range(i + 1, len(boxes)):
            if j in skip_indices:
                continue
            box2 = boxes[j]
            class2, confidence2, x2_center, y2_center, w2, h2 = box2[:6]
            quadrant2 = get_quadrant(x2_center, y2_center)

            # 只考虑不同象限的框进行合并
            if quadrant1 != quadrant2:
                # 判断是否左右相邻
                horizontal_distance = abs((x1_center + w1 / 2) - (x2_center - w2 / 2))
                vertical_alignment = abs(y1_center - y2_center)
                if vertical_alignment < threshold and horizontal_distance < threshold:
                    merged_box = merge_boxes(box1, box2)
                    merged_boxes.append(merged_box)
                    skip_indices.add(i)
                    skip_indices.add(j)
                    merged = True
                    break

                # 判断是否上下相邻
                vertical_distance = abs((y1_center + h1 / 2) - (y2_center - h2 / 2))
                horizontal_alignment = abs(x1_center - x2_center)
                if horizontal_alignment < threshold and vertical_distance < threshold:
                    merged_box = merge_boxes(box1, box2)
                    merged_boxes.append(merged_box)
                    skip_indices.add(i)
                    skip_indices.add(j)
                    merged = True
                    break

        if not merged:
            merged_boxes.append(box1)

    return merged_boxes
