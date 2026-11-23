"""Classical document detector: brightness and edge candidates."""

from __future__ import annotations

import time

import cv2
import numpy as np

from .config import (
    APPROX_EPSILON,
    CANNY_SIGMA,
    GAUSSIAN_KERNEL,
    MAX_AREA_RATIO,
    MAX_DIMENSION,
    MIN_AREA_RATIO,
)
from .geometry import is_convex_quad, order_corners, polygon_area, warp_from_corners


def _resize(image: np.ndarray, max_dimension: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, max_dimension / max(height, width))
    if scale == 1.0:
        return image, 1.0
    resized = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def _quad_from_contour(contour: np.ndarray, image_area: float) -> np.ndarray | None:
    area = cv2.contourArea(contour)
    ratio = area / image_area
    if not (MIN_AREA_RATIO <= ratio <= MAX_AREA_RATIO):
        return None

    perimeter = cv2.arcLength(contour, True)
    approximation = cv2.approxPolyDP(contour, APPROX_EPSILON * perimeter, True)
    if len(approximation) == 4 and is_convex_quad(approximation):
        return approximation.reshape(4, 2).astype("float32")

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype("float32")
    box_area = polygon_area(box)
    if MIN_AREA_RATIO <= box_area / image_area <= MAX_AREA_RATIO and is_convex_quad(box):
        return box
    return None


def _best_quad(mask: np.ndarray, image_area: float) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours[:8]:
        quad = _quad_from_contour(contour, image_area)
        if quad is not None:
            return quad
    return None


def _brightness_candidate(gray: np.ndarray) -> np.ndarray | None:
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return _best_quad(mask, float(gray.shape[0] * gray.shape[1]))


def _edge_candidate(gray: np.ndarray) -> np.ndarray | None:
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, 0)
    median = float(np.median(blurred))
    lower = int(max(0, (1.0 - CANNY_SIGMA) * median))
    upper = int(min(255, (1.0 + CANNY_SIGMA) * median))
    edges = cv2.Canny(blurred, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return _best_quad(edges, float(gray.shape[0] * gray.shape[1]))


def _candidate_score(gray: np.ndarray, quad: np.ndarray) -> float:
    """Rank a candidate by bright-inside contrast and plausible area.

    The page is the largest bright object, so a correct quadrilateral has a
    clearly brighter interior than its surroundings. Weighting by the square
    root of the area prevents small bright patches from winning.
    """
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(quad).astype(np.int32)], 255)
    inside = mask > 0
    if inside.sum() < 16 or (~inside).sum() < 16:
        return -1.0
    contrast = float(gray[inside].mean()) - float(gray[~inside].mean())
    area_ratio = float(inside.mean())
    return contrast * (max(area_ratio, 0.0) ** 0.5)


def detect_document(image: np.ndarray, max_dimension: int = MAX_DIMENSION) -> dict:
    """Detect the largest document quadrilateral and rectify it.

    Two independent candidate sources are used: an Otsu brightness mask (robust
    for a light page on a darker background) and a Canny edge map (robust when
    the page touches a bright region). Candidates are ranked by interior
    brightness contrast weighted by area, which rejects background texture and
    partial-page blobs.
    """
    if image is None or image.size == 0:
        raise ValueError("empty image")
    started = time.perf_counter()
    resized, scale = _resize(image, max_dimension)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    candidates = []
    for method, finder in (("brightness", _brightness_candidate), ("edges", _edge_candidate)):
        quad = finder(gray)
        if quad is not None:
            candidates.append((method, quad, _candidate_score(gray, quad)))

    if not candidates:
        return {
            "corners": None,
            "warped": None,
            "method": "none",
            "confidence": 0.0,
            "area_ratio": 0.0,
            "runtime_ms": (time.perf_counter() - started) * 1000,
        }

    method, quad, score = max(candidates, key=lambda item: item[2])
    corners = order_corners(quad / scale)
    warped = warp_from_corners(image, corners)
    area_ratio = polygon_area(corners) / float(image.shape[0] * image.shape[1])

    return {
        "corners": corners,
        "warped": warped,
        "method": method,
        "confidence": float(min(1.0, max(0.0, score) / 120.0)),
        "area_ratio": float(area_ratio),
        "runtime_ms": (time.perf_counter() - started) * 1000,
    }


def draw_detection(image: np.ndarray, corners: np.ndarray | None, colour=(60, 220, 120)) -> np.ndarray:
    """Draw the detected quadrilateral on a copy of the image."""
    canvas = image.copy()
    if corners is not None:
        points = np.round(order_corners(corners)).astype(np.int32)
        cv2.polylines(canvas, [points], True, colour, 3, cv2.LINE_AA)
        for point in points:
            cv2.circle(canvas, tuple(point), 5, colour, -1, cv2.LINE_AA)
    return canvas
