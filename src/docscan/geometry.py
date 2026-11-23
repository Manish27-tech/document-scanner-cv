"""Geometry helpers: corner ordering, homography, warping and IoU."""

from __future__ import annotations

import cv2
import numpy as np


def order_corners(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype="float32").reshape(4, 2)
    ordered = np.zeros((4, 2), dtype="float32")
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).ravel()
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def quad_dimensions(points: np.ndarray) -> tuple[int, int]:
    """Estimate the width and height of an ordered quadrilateral."""
    pts = order_corners(points)
    width = int(
        max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3]))
    )
    height = int(
        max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2]))
    )
    return max(width, 1), max(height, 1)


def warp_from_corners(
    image: np.ndarray, corners: np.ndarray, out_size: tuple[int, int] | None = None
) -> np.ndarray:
    """Perspective-rectify the quadrilateral described by ``corners``."""
    corners = order_corners(corners)
    if out_size is None:
        out_size = quad_dimensions(corners)
    width, height = out_size
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(corners.astype("float32"), destination)
    return cv2.warpPerspective(image, matrix, (width, height))


def quad_mask(shape: tuple[int, int], corners: np.ndarray) -> np.ndarray:
    """Binary mask filled by the quadrilateral."""
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(corners).astype(np.int32)], 255)
    return mask


def quad_iou(corners_a: np.ndarray, corners_b: np.ndarray, shape: tuple[int, int]) -> float:
    """Intersection-over-union between two quadrilaterals."""
    first = quad_mask(shape, corners_a) > 0
    second = quad_mask(shape, corners_b) > 0
    union = np.logical_or(first, second).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(first, second).sum() / union)


def corner_error(predicted: np.ndarray, truth: np.ndarray, shape: tuple[int, int]) -> float:
    """Mean corner distance normalised by the image diagonal."""
    predicted = order_corners(predicted)
    truth = order_corners(truth)
    diagonal = float(np.hypot(shape[0], shape[1]))
    return float(np.linalg.norm(predicted - truth, axis=1).mean() / diagonal)


def polygon_area(points: np.ndarray) -> float:
    """Shoelace area of a polygon."""
    pts = np.asarray(points, dtype="float32").reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def quad_area(points: np.ndarray) -> float:
    """Alias used by tests and reports for a four-point polygon."""
    return polygon_area(points)


def is_convex_quad(points: np.ndarray) -> bool:
    """Return True when four ordered points form a convex quadrilateral."""
    pts = np.asarray(points, dtype="float32").reshape(4, 2)
    signs = []
    for index in range(4):
        a = pts[index]
        b = pts[(index + 1) % 4]
        c = pts[(index + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        signs.append(np.sign(cross))
    return len(set(signs)) == 1
