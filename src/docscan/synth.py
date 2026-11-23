"""Synthetic document scenes with exact corner ground truth.

A page image is rendered with text, a table and a header, then projected onto a
random background with a homography. Conditions (blur, noise, shadow, rotation,
low contrast) are applied after compositing. The generator is deterministic
given a seed and does not require any external dataset.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import CANVAS_SIZE, CONDITIONS, PAGE_SIZE


def _font(size: int) -> ImageFont.FreeTypeFont:
    """Load a bundled DejaVu font through matplotlib, with a PIL fallback."""
    try:
        from matplotlib import font_manager

        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
    except Exception:  # pragma: no cover - fallback for minimal environments
        return ImageFont.load_default()


def make_page(seed: int = 0, size: tuple[int, int] = PAGE_SIZE) -> np.ndarray:
    """Render a synthetic document page (BGR uint8)."""
    rng = np.random.default_rng(seed)
    width, height = size
    paper = int(rng.integers(242, 255))
    image = Image.new("RGB", (width, height), (paper, paper, paper - int(rng.integers(0, 6))))
    draw = ImageDraw.Draw(image)
    ink = (32, 38, 48)
    accent = (28, 78, 140)

    draw.rectangle([0, 0, width, 88], fill=accent)
    draw.text((36, 24), "DOCSCAN LABORATORY", fill=(255, 255, 255), font=_font(30))
    draw.text((36, 106), "Field Report — Document Capture", fill=ink, font=_font(24))
    draw.text((36, 140), f"Reference DS-{seed:04d} · Synthetic specimen", fill=(95, 105, 120), font=_font(15))
    draw.line([36, 168, width - 36, 168], fill=(170, 178, 188), width=2)

    y = 190
    for _ in range(2):
        draw.text((36, y), "Abstract", fill=accent, font=_font(17))
        y += 26
        for _ in range(4):
            line_width = int(rng.integers(int(width * 0.55), width - 60))
            draw.rounded_rectangle([36, y, line_width, y + 10], radius=4, fill=(120, 128, 138))
            y += 24
        y += 12

    draw.text((36, y), "Measurements", fill=accent, font=_font(17))
    y += 30
    rows, cols = 4, 4
    table_width = width - 72
    table_height = 150
    cell_w = table_width / cols
    cell_h = table_height / rows
    for row in range(rows + 1):
        draw.line([36, y + row * cell_h, 36 + table_width, y + row * cell_h], fill=(150, 158, 170), width=1)
    for col in range(cols + 1):
        draw.line([36 + col * cell_w, y, 36 + col * cell_w, y + table_height], fill=(150, 158, 170), width=1)
    for row in range(rows):
        for col in range(cols):
            draw.text(
                (44 + col * cell_w, y + 10 + row * cell_h),
                f"{rng.integers(10, 99)}.{rng.integers(0, 9)}",
                fill=ink,
                font=_font(13),
            )
    y += table_height + 34

    for _ in range(5):
        line_width = int(rng.integers(int(width * 0.45), width - 60))
        draw.rounded_rectangle([36, y, line_width, y + 10], radius=4, fill=(135, 142, 152))
        y += 24

    draw.rectangle([width - 170, height - 150, width - 36, height - 36], outline=(190, 196, 205), width=2)
    draw.ellipse([width - 150, height - 130, width - 56, height - 56], outline=accent, width=3)
    draw.text((width - 144, height - 52), "APPROVED", fill=accent, font=_font(12))
    draw.line([36, height - 60, 260, height - 60], fill=(170, 178, 188), width=1)
    draw.text((36, height - 50), "DocScan QA · synthetic", fill=(130, 138, 150), font=_font(12))

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def make_background(
    size: tuple[int, int] = CANVAS_SIZE,
    seed: int = 0,
    brightness: int | None = None,
) -> np.ndarray:
    """Create a textured, darker background so the page is separable."""
    rng = np.random.default_rng(seed)
    width, height = size
    base = int(rng.integers(35, 140)) if brightness is None else brightness
    gradient = np.linspace(0.75, 1.15, width).reshape(1, width)
    canvas = np.clip(base * gradient, 0, 255) * np.ones((height, 1))
    canvas = canvas[..., None].repeat(3, axis=2)

    for _ in range(14):
        color = tuple(int(v) for v in rng.integers(20, 190, size=3))
        center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        axes = (int(rng.integers(40, 260)), int(rng.integers(40, 260)))
        overlay = canvas.astype(np.uint8).copy()
        cv2.ellipse(overlay, center, axes, int(rng.integers(0, 180)), 0, 360, color, -1)
        canvas = cv2.addWeighted(canvas, 0.82, overlay.astype(np.float64), 0.18, 0)

    noise = rng.normal(0, 6, canvas.shape)
    return np.clip(canvas + noise, 0, 255).astype(np.uint8)


def project_page(
    page: np.ndarray,
    canvas: np.ndarray,
    seed: int = 0,
    angle: float = 0.0,
    jitter: float = 0.06,
    shadow: float = 0.22,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a page onto a canvas and return the composite and ground-truth corners."""
    rng = np.random.default_rng(seed)
    canvas_height, canvas_width = canvas.shape[:2]
    page_height, page_width = page.shape[:2]

    scale = 0.68 * canvas_height / page_height
    rect_w, rect_h = page_width * scale, page_height * scale
    rect = np.array(
        [
            [-rect_w / 2, -rect_h / 2],
            [rect_w / 2, -rect_h / 2],
            [rect_w / 2, rect_h / 2],
            [-rect_w / 2, rect_h / 2],
        ],
        dtype="float32",
    )
    radians = np.deg2rad(angle)
    rotation = np.array(
        [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]],
        dtype="float32",
    )
    rect = rect @ rotation.T + np.array([canvas_width / 2, canvas_height / 2], dtype="float32")

    perturb = rng.uniform(-jitter, jitter, size=(4, 2)) * np.array([rect_w, rect_h])
    destination = rect + perturb

    margin = 18
    minimum = destination.min(axis=0)
    maximum = destination.max(axis=0)
    shift = np.zeros(2)
    shift[0] = margin - minimum[0] if minimum[0] < margin else 0
    shift[1] = margin - minimum[1] if minimum[1] < margin else 0
    if maximum[0] + shift[0] > canvas_width - margin:
        shift[0] -= maximum[0] + shift[0] - (canvas_width - margin)
    if maximum[1] + shift[1] > canvas_height - margin:
        shift[1] -= maximum[1] + shift[1] - (canvas_height - margin)
    destination += shift
    destination = destination.astype("float32")

    source = np.array(
        [[0, 0], [page_width - 1, 0], [page_width - 1, page_height - 1], [0, page_height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(page, matrix, (canvas_width, canvas_height))
    mask = cv2.warpPerspective(
        np.full((page_height, page_width), 255, dtype=np.uint8),
        matrix,
        (canvas_width, canvas_height),
    )

    shadow_mask = np.roll(mask, (int(canvas_height * 0.012), int(canvas_width * 0.012)), axis=(0, 1))
    shadow_mask = cv2.GaussianBlur(shadow_mask, (0, 0), sigmaX=18)
    darkness = (shadow_mask.astype(np.float32) / 255.0) * shadow
    composite = canvas.astype(np.float32) * (1.0 - darkness[..., None])
    composite = composite.astype(np.uint8)
    composite[mask > 0] = warped[mask > 0]

    return composite, destination.astype("float32")


def apply_condition(
    image: np.ndarray,
    condition: str,
    severity: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Apply a named degradation after compositing."""
    rng = np.random.default_rng(seed + 991)
    result = image.copy()
    height, width = result.shape[:2]

    if condition == "blur":
        kernel = int(3 + 4 * severity) | 1
        result = cv2.GaussianBlur(result, (kernel, kernel), sigmaX=0.8 + 1.6 * severity)

    if condition == "noise":
        sigma = 6 + 26 * severity
        noise = rng.normal(0, sigma, result.shape)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if condition in {"shadow", "lowcontrast"}:
        strength = 0.25 + 0.35 * severity if condition == "shadow" else 0.12
        gradient = np.linspace(1.0 - strength, 1.0, width).reshape(1, width)
        if rng.random() < 0.5:
            gradient = gradient[:, ::-1]
        result = np.clip(result.astype(np.float32) * gradient[..., None], 0, 255).astype(np.uint8)

    if condition == "lowcontrast":
        result = cv2.addWeighted(result, 0.78, np.full_like(result, 165), 0.22, 0)

    return result


def generate_sample(
    sample_id: int,
    condition: str = "clean",
    severity: float = 1.0,
    seed: int | None = None,
    canvas_size: tuple[int, int] = CANVAS_SIZE,
    page_seed: int | None = None,
    angle_override: float | None = None,
) -> dict:
    """Generate one synthetic scene and its ground-truth metadata."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    rng = np.random.default_rng((seed if seed is not None else sample_id) + 7)
    page_seed = page_seed if page_seed is not None else sample_id

    angle = {
        "clean": rng.uniform(-4, 4),
        "rotate": rng.uniform(14, 24) * (1 if rng.random() < 0.5 else -1),
        "blur": rng.uniform(-6, 6),
        "noise": rng.uniform(-8, 8),
        "shadow": rng.uniform(-8, 8),
        "lowcontrast": rng.uniform(-12, 12),
    }[condition]
    if angle_override is not None:
        angle = float(angle_override)

    background_brightness = int(rng.integers(145, 185)) if condition == "lowcontrast" else None
    page = make_page(seed=page_seed)
    if condition == "lowcontrast":
        page = cv2.addWeighted(page, 0.9, np.full_like(page, 210), 0.1, 0)
    background = make_background(canvas_size, seed=sample_id + 31, brightness=background_brightness)
    shadow = 0.22 + (0.18 * severity if condition == "shadow" else 0.0)
    composite, corners = project_page(
        page, background, seed=sample_id, angle=angle, shadow=shadow
    )
    image = apply_condition(composite, condition, severity=severity, seed=sample_id)

    return {
        "sample_id": sample_id,
        "image": image,
        "corners": corners,
        "condition": condition,
        "severity": float(severity),
        "page_size": (page.shape[1], page.shape[0]),
        "canvas_size": (image.shape[1], image.shape[0]),
    }


def save_sample(sample: dict, path: str | Path) -> Path:
    """Write a generated scene to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sample["image"])
    return path
