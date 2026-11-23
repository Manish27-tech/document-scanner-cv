.PHONY: install evaluate samples test lint check serve clean

install:
	python -m pip install -r requirements-dev.txt

samples:
	python scripts/generate_samples.py

evaluate:
	python run.py evaluate

test:
	python -m pytest

lint:
	ruff check .

check:
	python run.py check

serve:
	streamlit run dashboard/app.py

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
