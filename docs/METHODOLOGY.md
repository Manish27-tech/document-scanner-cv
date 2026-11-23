# Methodology

## 1. Task

Given a photograph containing one document, return the four page corners in a
consistent order and a perspective-rectified page. The task is evaluated as a
geometry problem, not a pixel-perfect reconstruction problem: corner error and
quadrilateral IoU are the primary measures.

## 2. Synthetic scene renderer

A page is drawn with PIL at 700×1,000 px:

- coloured header and title;
- abstract lines with random lengths;
- a 4×4 measurement table;
- footer and an approval box.

The page is projected onto a 1,100×850 px textured background with a homography.
The destination corners are recorded as ground truth. Shadow is composited by
offsetting and blurring the page mask before blending. Degradations are applied
after compositing:

| Condition | Degradation |
| --- | --- |
| clean | mild rotation, no extra noise |
| rotate | ±14–24° page rotation |
| blur | Gaussian blur, severity-scaled kernel |
| noise | additive Gaussian noise |
| shadow | stronger offset shadow and illumination gradient |
| lowcontrast | bright background and reduced page contrast |

Generation is deterministic given a seed; the evaluation set is regenerated on
every run.

## 3. Candidate detection

Two independent sources are used because they fail differently:

1. **Brightness candidate.** Otsu threshold, morphological closing and opening.
   This works when the page is the largest bright region.
2. **Edge candidate.** Gaussian blur, median-adaptive Canny thresholds,
   dilation and closing. This works when brightness alone is ambiguous.

For each mask, external contours are sorted by area. Contours covering 8–97% of
the frame are approximated; a convex four-point approximation is accepted, or a
minimum-area rectangle is used as a fallback.

## 4. Candidate ranking

A candidate is scored by

```
score = (mean intensity inside − mean intensity outside) × √area_ratio
```

This favours the bright page and rejects small bright patches, background
texture and partial-page blobs. The highest-scoring candidate is ordered
top-left, top-right, bottom-right, bottom-left and warped with a perspective
transform sized from the average opposite edge lengths.

## 5. Metrics

- **Corner error:** mean Euclidean distance between predicted and ground-truth
  corners, normalised by the image diagonal.
- **Quad IoU:** intersection-over-union of the filled quadrilaterals.
- **Strict success:** corner error ≤ 0.05 and IoU ≥ 0.80.
- **Runtime:** wall-clock time per scene, including resize and warping.

The strict criterion prevents a detector from scoring well by matching area
while placing corners in the wrong order or systematically shrinking the quad.

## 6. Robustness sweep

Success is measured across rotation (0–25°), blur severity (0–2), noise severity
(0–2) and shadow strength (0–1.5), with eight repetitions per level. The sweep
is repeated from fixed seeds, so the curve can be regenerated exactly.

## 7. Reproducibility

- Corpus seed: 42.
- Detector parameters: `src/docscan/config.py`.
- Every artifact is listed with size and SHA-256 in `artifacts/manifest.json`.
- CI runs the full test suite on synthetic mini-batches and does not require a
  display server.
