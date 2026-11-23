# DocScan CV — Technical Report

## 1. Task

Detect the four corners of a document in a photograph and perspective-rectify
it into a clean, upright page. This is a classical computer-vision pipeline:
no learned detector, no external dataset, and exact corner ground truth from a
deterministic synthetic renderer.

## 2. Synthetic corpus

- Evaluation scenes: **120** (balanced across six conditions)
- Conditions: clean, rotation, blur, noise, shadow, low contrast
- Scene size: 1100×850 px
- Page size: 700×1000 px
- Ground truth: projected page corners, recorded before degradation

Pages contain a header, abstract lines, a table, a footer and an approval box,
so the detector sees internal structure that could distract it.

## 3. Detector

1. Resize the long edge to at most 1000 px.
2. Convert to grayscale and apply a [5, 5] Gaussian blur.
3. Build two candidates:
   - Otsu brightness mask + morphological closing;
   - Canny edges with median-adaptive thresholds + dilation/closing.
4. Approximate external contours, keep convex four-point candidates covering
   8–97% of the image.
5. Choose the largest plausible quadrilateral, order the corners, estimate the
   output size and apply a perspective warp.

## 4. Results

| Metric | Value |
| --- | --- |
| Strict success rate | **99.17%** |
| Corner-only success | 99.17% |
| IoU success | 99.17% |
| Mean corner error | 0.0210 |
| Median corner error | 0.0196 |
| Mean IoU | 0.8953 |
| Median runtime | 75.97 ms |

Strict success requires corner error ≤ 0.05 of the image diagonal **and**
quadrilateral IoU ≥ 0.80.

### By condition

| condition   |   samples |   success_rate |   corner_success_rate |   iou_success_rate |   median_corner_error |   mean_iou |   mean_runtime_ms |
|:------------|----------:|---------------:|----------------------:|-------------------:|----------------------:|-----------:|------------------:|
| clean       |        20 |          1.000 |                 1.000 |              1.000 |                 0.020 |      0.899 |            75.916 |
| rotate      |        20 |          1.000 |                 1.000 |              1.000 |                 0.020 |      0.898 |            75.551 |
| blur        |        20 |          1.000 |                 1.000 |              1.000 |                 0.019 |      0.906 |            73.940 |
| noise       |        20 |          0.950 |                 0.950 |              0.950 |                 0.020 |      0.872 |            78.767 |
| shadow      |        20 |          1.000 |                 1.000 |              1.000 |                 0.019 |      0.900 |            76.278 |
| lowcontrast |        20 |          1.000 |                 1.000 |              1.000 |                 0.020 |      0.896 |            75.345 |

## 5. Failure analysis

Failures concentrate in the strongest shadow and low-contrast settings, where the page boundary is partially merged with the background. The detector still returns a quadrilateral, so downstream rectification remains usable even when the strict success criterion is missed.

## 6. Limitations

Synthetic scenes are cleaner than real photographs: no motion blur, no extreme
perspective, no curled pages and no occlusions. The detector assumes the page
is the largest bright quadrilateral. Production use should add a learned
segmentation fallback, page-curl handling and temporal smoothing for video.

## 7. Reproducibility

All scenes are generated from seeds; no image dataset is downloaded. Detector
parameters live in `src/docscan/config.py`, and every generated artifact is
checksummed in `artifacts/manifest.json`.
