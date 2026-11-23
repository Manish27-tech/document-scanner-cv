import numpy as np

from docscan.geometry import (
    corner_error,
    is_convex_quad,
    order_corners,
    quad_area,
    quad_dimensions,
    quad_iou,
    quad_mask,
    warp_from_corners,
)


def test_order_corners_returns_tl_tr_br_bl():
    points = np.array([[90, 90], [10, 10], [90, 10], [10, 90]], dtype="float32")
    ordered = order_corners(points)
    assert ordered[0].tolist() == [10, 10]
    assert ordered[1].tolist() == [90, 10]
    assert ordered[2].tolist() == [90, 90]
    assert ordered[3].tolist() == [10, 90]


def test_quad_dimensions_and_area():
    points = np.array([[0, 0], [100, 0], [100, 60], [0, 60]], dtype="float32")
    assert quad_dimensions(points) == (100, 60)
    assert quad_area(points) == 6000.0


def test_iou_perfect_and_disjoint():
    first = np.array([[0, 0], [50, 0], [50, 50], [0, 50]], dtype="float32")
    assert quad_iou(first, first, (100, 100)) == 1.0
    second = np.array([[60, 60], [90, 60], [90, 90], [60, 90]], dtype="float32")
    assert quad_iou(first, second, (100, 100)) == 0.0


def test_corner_error_zero_for_identical():
    points = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype="float32")
    assert corner_error(points, points, (100, 100)) == 0.0


def test_warp_returns_expected_shape():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    corners = np.array([[10, 10], [180, 10], [180, 190], [10, 190]], dtype="float32")
    warped = warp_from_corners(image, corners)
    assert warped.shape[0] == 180
    assert warped.shape[1] == 170


def test_mask_and_convexity():
    points = np.array([[0, 0], [50, 0], [50, 50], [0, 50]], dtype="float32")
    assert quad_mask((60, 60), points).sum() > 0
    assert is_convex_quad(points)
    assert not is_convex_quad(np.array([[0, 0], [50, 0], [0, 50], [50, 50]], dtype="float32"))
