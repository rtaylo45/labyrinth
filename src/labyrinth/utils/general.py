#!/usr/bin/env python
# Vidrovr Inc.

import signal

import numpy as np

from labyrinth.utils.exceptions import TimeoutException, timeout_handler


def with_timeout(seconds, func, *args, **kwargs):
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        result = func(*args, **kwargs)
    except TimeoutException as e:
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)

    return result


def calculate_iou(box_a, box_b):
    """Calculates the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box_a (np.ndarray): Coordinates of the first bounding box in the format [x1, y1, x2, y2].
        box_b (np.ndarray): Coordinates of the second bounding box in the format [x1, y1, x2, y2].

    Returns:
        float: The IoU value between the two bounding boxes.
    """
    x_a = np.maximum(box_a[0], box_b[0])
    y_a = np.maximum(box_a[1], box_b[1])
    x_b = np.minimum(box_a[2], box_b[2])
    y_b = np.minimum(box_a[3], box_b[3])

    intersection_area = np.maximum(0, x_b - x_a) * np.maximum(0, y_b - y_a)

    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = box_a_area + box_b_area - intersection_area

    iou = intersection_area / union_area
    return iou
