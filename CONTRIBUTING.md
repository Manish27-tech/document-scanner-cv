# Contributing

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python run.py evaluate
python -m pytest
```

## Before a pull request

1. Run `ruff check .` and `python -m pytest`.
2. Keep synthetic generation deterministic; changing seeds invalidates the
   reported metrics and pipeline card.
3. If detector behaviour changes, regenerate artifacts with
   `python run.py evaluate` and commit the updated figures, reports and
   manifest.
4. Do not commit generated scenes under `data/generated/`.

## Style

- Python 3.10+, 100-character lines, ruff defaults.
- Type hints on public functions.
- Tests must not require network access or a display server.

## Issues

Include the command, Python/OpenCV versions and the full traceback. For
detection issues, attach the input image and the overlay produced by
`python run.py scan`.
