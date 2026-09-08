from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MOVIES_FILE = DATA_DIR / "tmdb_5000_movies.csv"
CREDITS_FILE = DATA_DIR / "tmdb_5000_credits.csv"
