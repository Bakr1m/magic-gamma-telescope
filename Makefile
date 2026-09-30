.PHONY: install install-train install-dev test lint train clean

install:            ## Install runtime deps into .venv
	.venv/bin/pip install -r requirements.txt

install-train: install  ## + TF/imbalanced-learn/plotting (grid search needs these)
	.venv/bin/pip install -r requirements-train.txt

install-dev: install-train  ## + dev tools (pytest, ruff)
	.venv/bin/pip install -r requirements-dev.txt

test:               ## Run the test suite
	.venv/bin/python -m pytest tests/ -q

lint:               ## Lint src/tests with ruff
	.venv/bin/ruff check src tests

train:              ## Run the 54-config MLP grid search (saves models/mlp_best.keras + report)
	.venv/bin/python -m src.train

clean:              ## Remove caches (never touches data/, models/)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
