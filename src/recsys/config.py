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

# --- Day 3: baselines ---
RESULTS_DIR = REPORTS_DIR / "results"          # small JSON/markdown results (committed)
RUNS_DIR = DATA_DIR / "runs"                    # per-user metrics, cached matrices (git-ignored)
HEAD_SHARE = 0.10           # "head" = top 10% most-rated movies in training (EDA: ~79% of ratings)
COLD_SEED_COUNTS = (0, 1, 3, 5)
TUNE_USERS = 30000          # val users sampled for tuning (seeded); final test uses ALL users
KNN_NEIGHBORS = 200         # TF-IDF item-kNN: neighbours kept per movie
# Grid tuned on validation. Defaults below are used if you skip the tuning stage.
POP_MODES = ("count", "liked")
TFIDF_GRID = [{"weighting": w, "max_history": h}
              for w in ("liked", "centered") for h in (None, 50)]
TFIDF_DEFAULT = {"weighting": "liked", "max_history": None}

# --- Day 4: sentence embeddings + FAISS ---
EMB_DIR = DATA_DIR / "embeddings"               # <key>.npy / .ids.npy / .json / .faiss (git-ignored)
# Two small (384-dim) encoders compared on the SAME item documents as TF-IDF.
#   minilm    : 6 layers, trained on 1B sentence pairs; the standard fast baseline.
#   bge-small : 12 layers, contrastively trained for retrieval; stronger on MTEB, ~2x slower.
# bge wants an instruction prefix on short search QUERIES (Day 6 text onboarding),
# never on the documents themselves. MiniLM uses no prefix.
EMB_MODELS = {
    "minilm": {"hf_name": "sentence-transformers/all-MiniLM-L6-v2", "query_prefix": ""},
    "bge-small": {"hf_name": "BAAI/bge-small-en-v1.5",
                  "query_prefix": "Represent this sentence for searching relevant passages: "},
}
EMB_BATCH_SIZE = 64          # safe on an 8 GB CPU laptop; raise to 256 on a GPU
# Day 4 run 1 (centroid only, raw vectors) lost clearly to TF-IDF; 'liked' beat 'centered'
# weights for both encoders. Grid now: scoring method x mean-centering x history length.
#   centroid = cos(user mean vector, movie)  |  knn = same 200-neighbour item-kNN as TF-IDF
EMB_GRID = [{"method": m, "center": c, "max_history": h}
            for m in ("centroid", "knn") for c in (False, True) for h in (None, 50)]
EMB_KNN_NEIGHBORS = KNN_NEIGHBORS
SANITY_MOVIES = [("Toy Story", 1995), ("The Matrix", 1999), ("Titanic", 1997),
                 ("Spirited Away", 2001), ("The Dark Knight", 2008), ("The Notebook", 2004),
                 ("Alien", 1979), ("Inception", 2010)]

# --- Day 5: warm-user profiles, candidate recall, Flat vs HNSW ---
# Profile = normalise(Σ w(u,i)·v_i) on the Day 4 encoder + mean-centering (read from day04 json).
#   weighting : liked (1 per movie rated >= 4)  |  graded (4★ -> 1, 4.5★ -> 1.5, 5★ -> 2)
#   half_life : recency decay in liked-movie positions (None = no decay; 10 = the 10th most
#               recent liked movie counts half). Positions, not days (bulk rating sessions).
PROFILE_GRID = [{"weighting": w, "half_life": h}
                for w in ("liked", "graded") for h in (None, 3, 5, 10, 25, 50, 100)]
CANDIDATE_BUDGETS = (50, 100, 200, 500)       # stage-1 sizes: recall@N of the candidate set
CANDIDATE_MIX_BUDGET = 200                    # Day 8 plan: FAISS top-200 + popular fill
CANDIDATE_MIX_SHARES = (1.0, 0.75, 0.5, 0.25, 0.0)   # share of the 200 taken from embeddings
HNSW_M = (8, 16, 32, 64)                      # graph links per movie
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = (16, 32, 64, 128, 256, 512)  # query-time candidate list (FAISS uses >= k)
ANN_KS = (10, 200)                            # top-10 (a recommendation list), top-200 (candidates)
ANN_QUERIES = 2000                            # real user-profile queries from validation users
ANN_TARGET_RECALL = 0.95                      # "good enough" operating point for the API
