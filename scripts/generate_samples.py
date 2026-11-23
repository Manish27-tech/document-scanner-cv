"""Generate the small committed sample gallery with corner labels."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from docscan.config import CONDITIONS, SAMPLES_DIR  # noqa: E402
from docscan.synth import generate_sample, save_sample  # noqa: E402


def main() -> int:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(12):
        condition = CONDITIONS[index % len(CONDITIONS)]
        sample = generate_sample(
            sample_id=202600 + index,
            condition=condition,
            severity=1.0,
            seed=202600 + index,
        )
        name = f"{condition}_{index:03d}.png"
        save_sample(sample, SAMPLES_DIR / name)
        corners = sample["corners"].reshape(-1).tolist()
        rows.append(
            {
                "image": name,
                "sample_id": sample["sample_id"],
                "condition": condition,
                "severity": sample["severity"],
                **{f"c{position}": round(float(value), 2) for position, value in enumerate(corners)},
            }
        )
    pd.DataFrame(rows).to_csv(SAMPLES_DIR.parent / "samples.csv", index=False)
    print(f"wrote {len(rows)} samples to {SAMPLES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
