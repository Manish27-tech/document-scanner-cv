# Deployment

## Static results page (GitHub Pages)

The pipeline writes `docs/index.html`, `docs/RESULTS.md` and copies figures to
`docs/figures/`. In the repository settings:

1. **Settings → Pages**
2. **Deploy from a branch**
3. Branch `main`, folder `/docs`
4. Save; the page is served at `https://manish27-tech.github.io/document-scanner-cv/`

`docs/.nojekyll` disables Jekyll processing.

## Streamlit dashboard

```bash
pip install -r requirements-dev.txt
python run.py evaluate
streamlit run dashboard/app.py
```

Tabs: overview and committed samples, live detector (upload or sample),
evaluation figures and per-image table, robustness curves.

## Single-image CLI

```bash
python run.py scan photo.jpg --out artifacts/scan
```

Outputs `overlay.png` (detected quad) and `rectified.png` (warped page), plus a
JSON summary with method, confidence, area ratio, runtime, corners and output
size.

## Batch use

```python
import cv2
from docscan.detect import detect_document
from docscan.geometry import warp_from_corners

image = cv2.imread("photo.jpg")
result = detect_document(image)
if result["warped"] is not None:
    cv2.imwrite("page.png", result["warped"])
```

## Operational notes

- The detector expects one document, light on a darker background.
- Reject scans where no quad is found or the area ratio is implausible.
- Keep the original image alongside the rectified page for auditability.
- Captured documents may contain personal data: encrypt at rest, restrict
  access, redact before logging and delete on a defined schedule.
- For real deployments, add page-curl handling, multi-document support and a
  learned segmentation fallback.
