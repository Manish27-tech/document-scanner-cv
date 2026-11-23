# Design Decisions

## Why a synthetic corpus?

Corner labels for real document photos are slow to produce and ambiguous at the
pixel level. A deterministic renderer provides exact corners, controlled
degradations and repeatable experiments. The limitation is realism; the roadmap
explicitly lists page curl, occlusion and motion blur as future work.

## Why both brightness and edge candidates?

Otsu brightness is extremely reliable for a light page on a darker background,
but fails when a shadow splits the page mask. Canny edges survive lighting
changes, but can latch onto internal table lines or background texture. Running
both and ranking by bright-inside contrast combines their strengths.

## Why contrast × √area?

Area alone prefers the whole background; contrast alone prefers small bright
patches. Multiplying gives the correct page a high score while keeping the
scale meaningful across image sizes.

## Why corner error and IoU?

Corner error measures geometric accuracy; IoU measures area agreement. A quad
can have a good IoU while its corners are badly ordered, or accurate corners
while one edge drifts inward. Strict success requires both.

## Why normalise corner error by the diagonal?

Absolute pixels depend on image resolution. Normalising by the diagonal makes
the 5% threshold meaningful across the synthetic corpus and real photographs.

## Why no deep learning?

The page boundary is a strong, low-level geometric cue. A classical detector
runs in tens of milliseconds on a CPU, has no model weights, and is easy to
debug. A learned segmentation model is the right fallback for cluttered scenes,
but it is unnecessary for this benchmark and would break the dependency-light
reproducibility goal.

## Why a confidence score?

The dashboard needs a rough quality signal. Ranking score divided by 120 is a
bounded heuristic, not a calibrated probability; it is labelled "confidence"
but should not be interpreted as P(correct).

## Why commit sample scenes?

Reviewers can inspect the generator's output without running anything. The full
120-scene benchmark stays out of the repository and is regenerated from seeds.
