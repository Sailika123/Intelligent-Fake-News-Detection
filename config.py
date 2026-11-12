"""Shared project settings for data, models, and generated outputs."""
from pathlib import Path

# --- Paths -------------------------------------------------------------
DATA_DIR = "data"
TRUE_CSV = f"{DATA_DIR}/True.csv"
FAKE_CSV = f"{DATA_DIR}/Fake.csv"
CLEANED_DATASET_PATH = f"{DATA_DIR}/news_cleaned.csv"   # single EDA checkpoint, written once
MODEL_READY_PATH = f"{DATA_DIR}/news_model_ready.csv"    # single NLP-cleaned checkpoint, written once

OUTPUT_DIR = "outputs"
ARTIFACT_DIR = "artifacts"
LOG_DIR = "logs"

# --- Reproducibility -----------------------------------------------------
RANDOM_STATE = 42

# --- Preprocessing / modelling constants --------------------------------
MIN_CLEAN_WORD_COUNT = 3      # drop rows whose cleaned body has <= this many tokens
TEST_SIZE = 0.20

TFIDF_MAX_FEATURES = 15000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 3

VOCAB_SIZE = 20000
MAX_LEN = 180
EMBED_DIM = 64

PERFORMANCE_TARGET = 0.80      # minimum acceptable accuracy/precision/recall/F1

ABLATION_FRACTIONS = [1.00, 0.75, 0.50, 0.25, 0.10, 0.05]

# Subjects used for the cross-source generalisation robustness test
GENERALISATION_SEEN_SUBJECTS = ["News", "left-news", "politicsNews"]
GENERALISATION_UNSEEN_SUBJECTS = ["worldnews", "politics", "Government News", "US_News"]

def ensure_dirs():
    for d in (OUTPUT_DIR, ARTIFACT_DIR, LOG_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)
