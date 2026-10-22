from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ML_DIR = RAW_DIR / "ml-32m"
TMDB_DIR = RAW_DIR / "tmdb"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

MIN_USER_RATINGS = 20
MIN_ITEM_RATINGS = 20
SEED = 42

ML_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
TMDB_KAGGLE_SLUG = "alanvourch/tmdb-movies-daily-updates"
TMDB_POSTER_BASE = "https://image.tmdb.org/t/p/w342"
