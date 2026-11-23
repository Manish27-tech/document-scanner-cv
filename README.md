# DocScan CV — Document Detection & Rectification

[![CI](https://github.com/Manish27-tech/document-scanner-cv/actions/workflows/ci.yml/badge.svg)](https://github.com/Manish27-tech/document-scanner-cv/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4%2B-5C3EE8)](https://opencv.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

DocScan CV is a classical computer-vision pipeline that finds a document's
four corners in a photograph and perspective-rectifies it into an upright
page. It uses OpenCV and NumPy — no learned detector and no downloaded image
dataset. Ground truth comes from a deterministic synthetic scene renderer, so
every metric is reproducible.

## Results

| Metric | Value |
| --- | --- |
| Strict success | **99.17%** (119 / 120 scenes) |
| Mean IoU | **0.8953** |
| Median corner error | **0.0196** of the image diagonal |
| P95 corner error | 0.0221 |
| Mean runtime | **76 ms** per scene |

Strict success means corner error ≤ 5% of the image diagonal **and** quad IoU ≥ 0.80.

### Success by condition

| Condition | Success | Median corner error | Mean IoU |
| --- | --- | --- | --- |
| Clean | 100% | 0.020 | 0.879 |
| Rotation (±24°) | 100% | 0.020 | 0.897 |
| Blur | 100% | 0.020 | 0.922 |
| Noise | **95%** | 0.020 | 0.891 |
| Shadow | 100% | 0.019 | 0.896 |
| Low contrast | 100% | 0.019 | 0.888 |

The single failure is a heavily noised scene where the Otsu brightness mask
fragments and the detector latches onto a sub-region. Predicted-vs-truth
examples are in `artifacts/figures/08_failure_cases.png`.

## How it works

1. Resize the long edge to 1000 px.
2. Convert to grayscale and apply a 5×5 Gaussian blur.
3. Build two independent candidates:
   - **brightness**: Otsu threshold + morphological closing;
   - **edges**: median-adaptive Canny + dilation/closing.
4. Keep convex four-point contours covering 8–97% of the frame.
5. Rank candidates by interior brightness contrast × √area — this rejects
   background texture and partial-page blobs.
6. Order corners, estimate the output size, and apply a perspective warp.

## Quick start

```bash
git clone https://github.com/Manish27-tech/document-scanner-cv.git
cd document-scanner-cv
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt

python run.py evaluate          # 120 scenes, full artifacts, figures, report
python run.py check             # verify artifacts
python run.py scan path/to/photo.jpg --out artifacts/scan
python -m pytest
```

Sample gallery and labels:

```bash
python scripts/generate_samples.py
```

Launch the dashboard:

```bash
streamlit run dashboard/app.py
```

## Project layout

```
document-scanner-cv/
├── src/docscan/          # geometry, generator, detector, metrics, reports
├── dashboard/            # Streamlit viewer and upload demo
├── tests/                # geometry, generator, detector, pipeline tests
├── scripts/              # sample-gallery generator
├── data/samples/         # 12 committed synthetic scenes + corner labels
├── artifacts/            # figures, dashboard feed, manifest, examples
├── reports/              # technical report, pipeline card, metrics
├── docs/                 # static GitHub Pages results site
├── run.py                # evaluate / generate / scan / check CLI
└── .github/workflows/ci.yml
```

## Synthetic corpus

- 1,100×850 px scenes; 700×1,000 px pages
- Header, abstract lines, a 4×4 table, footer and approval box
- Random textured backgrounds and shadows
- Six conditions: clean, rotation, blur, noise, shadow, low contrast
- Exact corners recorded before degradation

Twelve scenes with labels are committed for inspection; the full 120-scene
evaluation set is regenerated deterministically from seeds.

## Why synthetic data?

Corner labels for real document photos are tedious and ambiguous. A
deterministic renderer gives exact geometry, controlled degradations and
repeatable experiments. The trade-off is realism: no page curl, motion blur,
occlusion or extreme perspective. A learned segmentation fallback is the
natural next step.

## Reproducibility

- Fixed seed (`42`) for corpus generation and robustness sweeps.
- All detector parameters in `src/docscan/config.py`.
- Every run writes `artifacts/manifest.json` with SHA-256 checksums.
- CI runs lint and the test suite on every push.

## Limitations

Single-document scenes, synthetic only, light page on a darker background.
Production use needs page-curl handling, multi-document support, temporal
smoothing for video and privacy controls for captured documents.

## License

MIT — see [LICENSE](LICENSE).

## Author

**Manish Pandey** · 2026
