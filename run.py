"""Command line entry point for DocScan CV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from docscan.config import ARTIFACTS, CONDITIONS, DATA_DIR, EVAL_SAMPLES  # noqa: E402
from docscan.detect import detect_document, draw_detection  # noqa: E402
from docscan.pipeline import run_pipeline, scan_summary  # noqa: E402
from docscan.synth import generate_sample, save_sample  # noqa: E402


def cmd_evaluate(args: argparse.Namespace) -> int:
    result = run_pipeline(
        n_samples=args.samples,
        robustness_reps=args.reps,
        show_progress=True,
    )
    metrics = result["metrics"]
    print(f"strict success : {metrics['success_rate']:.2%}")
    print(f"mean IoU       : {metrics['mean_iou']:.4f}")
    print(f"median error   : {metrics['median_corner_error']:.4f}")
    print(f"mean runtime   : {metrics['mean_runtime_ms']:.2f} ms")
    print(f"artifacts      : {len(result['manifest']['files'])} files")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    target = Path(args.out)
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(args.count):
        condition = CONDITIONS[index % len(CONDITIONS)]
        sample = generate_sample(
            sample_id=202600 + index,
            condition=condition,
            severity=1.0,
            seed=202600 + index,
        )
        name = f"{condition}_{index:03d}.png"
        save_sample(sample, target / name)
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
    labels = target.parent / "samples.csv"
    import pandas as pd

    pd.DataFrame(rows).to_csv(labels, index=False)
    print(f"generated {len(rows)} samples in {target}")
    print(f"labels: {labels}")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    import cv2

    image = cv2.imread(args.image)
    if image is None:
        print(f"could not read image: {args.image}")
        return 1
    result = detect_document(image)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    overlay = draw_detection(image, result["corners"])
    cv2.imwrite(str(output / "overlay.png"), overlay)
    if result["warped"] is not None:
        cv2.imwrite(str(output / "rectified.png"), result["warped"])
    print(json.dumps(scan_summary(result), indent=2))
    print(f"written to {output}")
    return 0


def cmd_check(_: argparse.Namespace) -> int:
    from docscan import __version__

    problems = []
    if not (ARTIFACTS / "figures").exists():
        problems.append("figures missing (run python run.py evaluate)")
    if not (ARTIFACTS / "dashboard" / "metrics.json").exists():
        problems.append("metrics missing (run python run.py evaluate)")
    if not (DATA_DIR / "samples").exists():
        problems.append("committed samples missing")
    print(f"docscan {__version__}")
    print(f"artifacts : {ARTIFACTS}")
    print(f"problems  : {problems or 'none'}")
    return 0 if not problems else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DocScan CV — document detection and rectification")
    sub = parser.add_subparsers(dest="command", required=True)

    evaluate = sub.add_parser("evaluate", help="run the full evaluation pipeline")
    evaluate.add_argument("--samples", type=int, default=EVAL_SAMPLES)
    evaluate.add_argument("--reps", type=int, default=8)
    evaluate.set_defaults(func=cmd_evaluate)

    generate = sub.add_parser("generate", help="write synthetic scenes and labels")
    generate.add_argument("--count", type=int, default=12)
    generate.add_argument("--out", type=str, default=str(DATA_DIR / "generated" / "images"))
    generate.set_defaults(func=cmd_generate)

    scan = sub.add_parser("scan", help="detect and rectify one image")
    scan.add_argument("image", type=str)
    scan.add_argument("--out", type=str, default=str(ARTIFACTS / "scan"))
    scan.set_defaults(func=cmd_scan)

    check = sub.add_parser("check", help="verify generated artifacts")
    check.set_defaults(func=cmd_check)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
