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

# --- Day 2: evaluation protocol (frozen once the split is written) ---
SPLIT_DIR = PROCESSED_DIR / "splits"
N_TEST = 10                 # last 10 ratings per user (by time) -> test
N_VAL = 10                  # the 10 before those -> validation (tuning only)
MIN_TRAIN_FOR_VAL = 10      # a user gets a val slice only if >= 10 train ratings remain
N_COLD_USERS = 5000         # users held out ENTIRELY from training (cold-start eval)
N_COLD_SEEDS = 5            # first 5 liked movies = onboarding seeds (eval at 0/1/3/5)
RELEVANCE_THRESHOLD = 4.0   # rating >= 4 counts as relevant
EVAL_KS = (5, 10, 20)
