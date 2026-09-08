# CineMatch: Movie Discovery & Performance Insights

CineMatch is a GitHub-ready machine learning project for content-based movie discovery and leakage-aware historical movie analytics. The repository is structured as an installable Python package, includes tested data utilities, automated CI, and documentation that clearly defines model scope and limitations.

> **Portfolio project, not a prediction product.** Historical financial and audience fields are treated carefully: final revenue is used only to form the retrospective label, while post-release popularity and voting variables are excluded from the historical model inputs.

## Highlights

- Content-based movie retrieval from genres, keywords, overview text, top-billed cast, and directors
- TF-IDF vectorization, cosine-similarity ranking, fuzzy title matching, and transparent similarity scores
- Optional quality filtering by source vote count and lightweight genre-signature diversification
- Safe parsing of nested TMDB JSON-like fields with reusable, tested utilities
- Chronological holdout baseline for retrospective performance analysis
- Installable Python package, command-line interface, unit tests, Ruff linting, Makefile, and GitHub Actions CI
- Model Card and Data Card documenting intended use, evaluation, risks, constraints, and data handling

## Repository Structure

```text
cinematch-movie-intelligence/
├── src/cinematch/
│   ├── config.py              # Shared project paths
│   ├── data.py                # Loading and safe metadata parsing
│   ├── recommender.py         # TF-IDF retrieval and re-ranking
│   ├── evaluation.py          # Offline genre-overlap evaluation
│   ├── analytics.py           # Leakage-aware historical baseline
│   └── cli.py                 # Command-line interface
├── notebooks/
│   ├── 01_recommendation_engine.ipynb
│   └── 02_performance_analysis.ipynb
├── tests/
│   └── test_data.py
├── .github/workflows/ci.yml   # Lint + test automation
├── data/                      # Local dataset only; ignored by Git
├── artifacts/                 # Generated outputs only; ignored by Git
├── README.md
├── MODEL_CARD.md
├── DATA_CARD.md
├── CONTRIBUTING.md
├── pyproject.toml
├── Makefile
└── LICENSE
```

## Data Setup

Download the TMDB 5000 Movie Dataset separately and place its two CSV files in `data/`:

```text
data/tmdb_5000_movies.csv
data/tmdb_5000_credits.csv
```

The dataset is deliberately excluded from version control. Read the original provider’s terms before using or redistributing it.

## Installation

```bash
git clone https://github.com/<your-github-username>/cinematch-movie-intelligence.git
cd cinematch-movie-intelligence
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the package plus development tools:

```bash
pip install -e ".[dev]"
```

## Commands

Recommend related movies:

```bash
python main.py recommend --title "Avatar" --top-k 10 --min-vote-count 100
```

The recommender uses diversity-aware results by default. To return the most purely similar titles without re-ranking:

```bash
python main.py recommend --title "Avatar" --top-k 10 --no-diversify
```

Evaluate broad content consistency with genre-overlap precision@K:

```bash
python main.py evaluate --sample-size 300 --top-k 10
```

Run the leakage-aware historical baseline with the default chronological holdout:

```bash
python main.py analyze --split chronological
```

Useful development commands:

```bash
make install
make lint
make test
```

## Recommendation Design

1. Movie and credit records are merged by the TMDB identifier.
2. Genres, keywords, cast, and crew fields are parsed safely.
3. Each movie receives a metadata profile that combines genres, keywords, overview text, top three cast members, and director.
4. `TfidfVectorizer` creates sparse feature vectors.
5. Cosine similarity retrieves candidate movies.
6. Optional vote-count filtering removes minimally supported catalog entries.
7. A lightweight greedy re-ranker reduces repeated identical genre signatures in the final list.

CineMatch is an **interpretable content-based system**. It does not claim to know a user’s personal taste because it has no user interaction data. It discovers catalog items with related metadata characteristics.

## Evaluation

The included metric is **genre-overlap precision@K**. A result is considered relevant when it shares at least one genre with the query film. This is useful for validating broad thematic consistency, but it is not a measure of personalized satisfaction, novelty, diversity, fairness, or commercial value.

For a real recommender deployment, evaluate with human relevance judgments or consented interaction data, and report ranking quality (for example NDCG), diversity, catalog coverage, novelty, latency, and performance across catalog segments.

## Historical ML Credibility

The historical analysis is intentionally framed as a **retrospective classification baseline**.

- The outcome label is whether historical revenue was at least historical budget.
- Revenue is used only for creating the label, never as a model input.
- Final popularity, vote count, and vote average are excluded because they are post-release signals.
- The default split is chronological: earlier releases train the model and more recent releases test it. This is more realistic than randomly mixing older and newer films.
- Results are not pre-release box-office forecasts, investment advice, or causal claims.

See `MODEL_CARD.md` for full scope, limitations, and responsible-use documentation.

## Quality Checks

GitHub Actions runs the following checks on pushes and pull requests to `main`:

```bash
ruff check src tests
pytest -q
```

## Future Work

- Compare TF-IDF retrieval against semantic plot embeddings
- Add hybrid retrieval with user feedback or ratings where properly licensed
- Add NDCG, diversity, coverage, and novelty evaluation
- Build a Streamlit interface or FastAPI endpoint
- Add artifact versioning and experiment tracking

## Author

**Hitika Ghanani**  
Aspiring Data Scientist / Machine Learning Engineer

Update this section with your GitHub, LinkedIn, and portfolio links before publishing.

## License

MIT License. See `LICENSE`.
