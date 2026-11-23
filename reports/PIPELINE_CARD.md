# Pipeline Card — DocScan CV

## Intended use

Educational and portfolio demonstration of classical document capture:
corner detection, quadrilateral validation, perspective rectification and
geometry-based evaluation.

## Inputs and outputs

- Input: BGR image containing one document on a contrasting background.
- Output: ordered four corners, rectified page, confidence and runtime.
- Ground truth required for evaluation: four page corners.

## Configuration

- Max long edge: 1000 px
- Gaussian kernel: [5, 5]
- Canny sigma: 0.66
- Minimum area ratio: 0.08
- Success thresholds: corner error ≤ 0.05, IoU ≥ 0.8

## Measured performance

Strict success **99.17%**, mean IoU **0.895**,
median corner error **0.0196** on
120 synthetic scenes.

## Limitations

Synthetic evaluation only. No page-curl, occlusion or multi-document handling.
The brightness candidate assumes a light page on a darker background.

## Ethical and operational notes

Document capture can expose personal data. Store rectified images securely,
redact sensitive fields before logging, and require user consent for uploads.
