# Contributing to CineMatch

Contributions are welcome. Please keep changes focused, reproducible, and respectful of dataset licensing.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
ruff check src tests
pytest -q
```

## Guidelines

- Do not commit TMDB CSV files, user data, credentials, or model artifacts.
- Add tests for new data-parsing or ranking behavior.
- Do not present offline genre-overlap results as personalized recommendation quality.
- Preserve the project’s leakage-aware distinction between historical analysis and real pre-release forecasting.
