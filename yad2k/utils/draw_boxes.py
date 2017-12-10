"""Draw predicted or ground truth boxes on input image."""

import colorsys
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from itertools import compress

import sys
sys.path.append("util")
from ml_utils import bb_intersection_over_union


def get_colors_for_classes(num_classes):
    """Return list of random colors for number of classes given."""
    # Use previously generated colors if num_classes is the same.
    if (hasattr(get_colors_for_classes, "colors") and
            len(get_colors_for_classes.colors) == num_classes):
        return get_colors_for_classes.colors

    hsv_tuples = [(float(x) / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    get_colors_for_classes.colors = colors  # Save colors for future calls.
    return colors

def draw_boxes(image, boxes, box_classes, class_names, scores=None):
    """Draw bounding boxes on image.

    Draw bounding boxes with class name and optional box score on image.

    Args:
        image: An `array` of shape (width, height, 3) with values in [0, 1].
        boxes: An `array` of shape (num_boxes, 4) containing box corners as
            (y_min, x_min, y_max, x_max).
        box_classes: A `list` of indicies into `class_names`.
        class_names: A `list` of `string` class names.
        `scores`: A `list` of scores for each box.

    Returns:
        A copy of `image` modified with given bounding boxes.
    """

    image = Image.fromarray(np.floor(image * 255 + 0.5).astype('uint8'))

    colors = get_colors_for_classes(len(class_names))

    for i, c in list(enumerate(box_classes)):
        box_class = class_names[c]
        box = boxes[i]
        color = colors[c]
        if isinstance(scores, np.ndarray):
            score = float(scores[i])
            label = '{} {:.2f}'.format(box_class, score)
            image = draw_box(image, box, box_class, c, score)
        else:
            label = '{}'.format(box_class)
            image = draw_box(image, box, box_class, color)

    return np.array(image)

def draw_box(image, box, box_class, color, score=None):

    font = ImageFont.truetype(
        font='data/font/FiraMono-Medium.otf',
        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    if score:
        label = '{} {:.2f}'.format(box_class, score)
    else:
        label = '{}'.format(box_class)

    draw = ImageDraw.Draw(image)
    label_size = draw.textsize(label, font)

    top, left, bottom, right = box
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
    right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
    print(label, (left, top), (right, bottom))

    if top - label_size[1] >= 0:
        text_origin = np.array([left, top - label_size[1]])
    else:
        text_origin = np.array([left, top + 1])

    # My kingdom for a good redistributable image drawing library.
    for i in range(thickness):
        draw.rectangle(
            [left + i, top + i, right - i, bottom - i], outline=color)
    draw.rectangle(
        [tuple(text_origin), tuple(text_origin + label_size)],
        fill=color)
    draw.text(text_origin, label, fill=(0, 0, 0), font=font)
    del draw

    return image

def draw_boxes_advanced(image, gt_boxes, gt_classes,
                        p_boxes, p_classes, p_scores, class_names,
                        score_threshold=0.0,  iou_threshold=0.5):

    image = Image.fromarray(np.floor(image * 255 + 0.5).astype('uint8'))

    tp_color, fn_color, fp_color  = get_colors_for_classes(3) #TP, FN, FP

    # Filter for boxes where score is above score_threshold
    boxes_classes_and_scores = list(zip(p_boxes, p_classes, p_scores))
    score_filter = np.array([True if score >= score_threshold \
                                  else False \
                                  for score in p_scores])
    filtered_data = list(compress(boxes_classes_and_scores, score_filter))

    # Edge case: if there are no boxes, increment nothing
    if len(gt_boxes) == 0:
        for box, box_class, score in filtered_data:
            image = draw_box(image, box, "fp-"+class_names[box_class], fp_color, score)

    # Edge case: if no boxes are discovered, increment false negatives
    elif len(filtered_data) == 0:
        for gt_box, gt_class in zip(gt_boxes, gt_classes):
            image = draw_box(image, gt_box, "fn-"+class_names[gt_class], fn_color)
    else:
        boxes_available = [1] * len(filtered_data)
        # Greedily assign predicted boxes to gt_boxes
        for gt_box, gt_class in zip(gt_boxes, gt_classes):
            detected = False
            # Predicted box must have high IOU with GT and be available
            for i, (p_box, p_class, p_score) in enumerate(filtered_data):
                iou = bb_intersection_over_union(gt_box, p_box)
                if iou >= iou_threshold and p_class == gt_class and boxes_available[i]:
                    boxes_available[i] = 0
                    detected = True
                    image = draw_box(image, p_box, "tp-"+class_names[p_class], tp_color, p_score)
                    break
            if not detected:
                image = draw_box(image, gt_box, "fn-"+class_names[gt_class], fn_color)
        for i, is_available in enumerate(boxes_available):
            if is_available:
                p_box, p_class, p_score = filtered_data[i]
                image = draw_box(image, p_box, "fp-"+class_names[p_class], fp_color, p_score)
    return np.array(image)
