.PHONY: setup test train train-fast serve clean help report

# Reproducibility: lock Python hash randomization
export PYTHONHASHSEED=42

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup:  ## Install dependencies
	pip install -e ".[dev]"

test:  ## Run test suite (57 tests)
	pytest tests/ -v

train:  ## Run full pipeline (all 4 models + baselines + backtest + report)
	python scripts/train_all.py

train-fast:  ## Run pipeline without classical models (~5 min vs ~30 min)
	python scripts/train_all.py --skip-classical

train-force:  ## Re-run everything from scratch, ignore cached artifacts
	python scripts/train_all.py --force

select:  ## Re-run selection and forecasting only (no retraining)
	python scripts/train_all.py --select-only

report:  ## Regenerate HTML report from existing artifacts
	python -c "import sys; sys.path.insert(0,'.'); from src.report import generate_report; generate_report()"

serve:  ## Start the API server on port 8000
	python scripts/serve.py

serve-dev:  ## Start API with auto-reload
	python scripts/serve.py --reload

clean:  ## Remove generated artifacts and processed data
	rm -rf artifacts/ data/processed/

lint:  ## Run basic code checks
	python -m py_compile config/settings.py
	python -m py_compile src/ingest.py
	python -m py_compile src/features.py
	python -m py_compile src/evaluate.py
	python -m py_compile src/baseline.py
	python -m py_compile src/report.py
	python -m py_compile api/app.py
	@echo "All modules compile successfully"
