"""
============================================================
Phase 2 NLP Pipeline — Step 02a: Text Preprocessing
============================================================
Project: Wisconsin Retirement AI
Script:  code/02_nlp_pipeline/02a_preprocessing.py
Author:  Khawaja Sazzad Ali
Date:    Summer 2026

PURPOSE:
    Clean and tokenize all 27,348 Wisconsin CFPB complaint
    narratives. Produces a single preprocessed corpus file
    that all three NLP tracks (A, B, C) consume.

    Preprocessing steps applied in order:
    1. Lowercase
    2. Remove CFPB anonymization tokens (xxxx patterns)
    3. Remove boilerplate submission language
    4. Remove punctuation and special characters
    5. Tokenize (split into words)
    6. Remove standard English stopwords
    7. Remove custom financial stopwords
    8. Lemmatize using spaCy
    9. Remove tokens shorter than 3 characters
    10. Rejoin tokens into clean string

    Also adds temporal era flag:
    - pre_2020: robo-advisor era (keyword set 1)
    - post_2020: LLM era (keyword set 2)

INPUT:
    data/processed/cfpb_wisconsin_with_county.parquet

OUTPUT:
    data/processed/cfpb_wisconsin_preprocessed.parquet
    docs/preprocessing_log.txt
============================================================
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ── 0. Paths ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
DOCS_DIR     = PROJECT_ROOT / "docs"

PROC_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH  = PROC_DIR / "cfpb_wisconsin_with_county.parquet"
OUTPUT_PATH = PROC_DIR / "cfpb_wisconsin_preprocessed.parquet"
LOG_PATH    = DOCS_DIR / "preprocessing_log.txt"

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

log("=" * 60)
log("Phase 2 NLP — Step 02a: Text Preprocessing")
log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 60)

# ── 1. Load spaCy ─────────────────────────────────────────────
log("\nLoading spaCy...")
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        log("  spaCy en_core_web_sm loaded")
        USE_SPACY = True
    except OSError:
        log("  en_core_web_sm not found — using regex lemmatization fallback")
        log("  To install: python -m spacy download en_core_web_sm")
        USE_SPACY = False
        nlp = None
except ImportError:
    log("  spaCy not installed — using regex lemmatization fallback")
    USE_SPACY = False
    nlp = None

# ── 2. Stopword lists ─────────────────────────────────────────
STANDARD_STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
    'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
    'did', 'will', 'would', 'could', 'should', 'may', 'might', 'shall',
    'can', 'this', 'that', 'these', 'those', 'it', 'its', 'they',
    'them', 'their', 'we', 'our', 'you', 'your', 'he', 'she', 'his',
    'her', 'i', 'my', 'me', 'not', 'no', 'nor', 'so', 'yet', 'both',
    'either', 'neither', 'each', 'any', 'all', 'few', 'more', 'most',
    'other', 'some', 'such', 'only', 'own', 'same', 'than', 'too',
    'very', 'just', 'also', 'up', 'out', 'if', 'about', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'between',
    'then', 'once', 'there', 'when', 'where', 'which', 'who', 'whom',
    'how', 'what', 'why', 'while', 'although', 'because', 'since',
    'until', 'unless', 'however', 'therefore', 'thus', 'hence',
}

# Financial stopwords — common but uninformative in this corpus
FINANCIAL_STOPWORDS = {
    'account', 'bank', 'company', 'payment', 'money', 'financial',
    'fund', 'service', 'card', 'credit', 'loan', 'debt', 'amount',
    'dollar', 'percent', 'fee', 'charge', 'month', 'year', 'day',
    'time', 'said', 'told', 'letter', 'call', 'called', 'spoke',
    'speak', 'contact', 'contacted', 'write', 'written', 'receive',
    'received', 'send', 'sent', 'make', 'made', 'take', 'taken',
    'come', 'came', 'get', 'got', 'give', 'given', 'know', 'knew',
    'go', 'went', 'see', 'seen', 'use', 'used', 'pay', 'paid',
    'try', 'tried', 'ask', 'asked', 'tell', 'told', 'want', 'wanted',
    'need', 'needed', 'also', 'would', 'could', 'one', 'two', 'three',
}

ALL_STOPWORDS = STANDARD_STOPWORDS | FINANCIAL_STOPWORDS

# ── 3. CFPB boilerplate patterns ──────────────────────────────
# Phrases that appear in virtually every complaint but carry no signal
BOILERPLATE_PATTERNS = [
    r'i am writing to',
    r'i would like to',
    r'please help me',
    r'thank you for your',
    r'i hope this letter',
    r'to whom it may concern',
    r'sincerely',
    r'consumer financial protection bureau',
    r'please investigate',
    r'i am a consumer',
]

# ── 4. Core preprocessing function ───────────────────────────
def preprocess_narrative(text: str) -> tuple[str, list[str]]:
    """
    Clean and tokenize one complaint narrative.

    Returns:
        (tokens_joined, token_list)
        tokens_joined: space-separated clean tokens (for LDA/embedding)
        token_list: list of individual tokens
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return "", []

    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Remove CFPB anonymization tokens
    # CFPB replaces names/numbers with xx... patterns
    text = re.sub(r'x{2,}', ' ', text)
    text = re.sub(r'\d{4,}', ' ', text)  # Long number strings

    # Step 3: Remove boilerplate
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, ' ', text)

    # Step 4: Remove punctuation and special characters
    # Keep apostrophes for contractions (handled below)
    text = re.sub(r"[^a-z\s']", ' ', text)
    text = re.sub(r"'s\b", '', text)    # Remove possessives
    text = re.sub(r"'", '', text)       # Remove remaining apostrophes

    # Step 5: Tokenize
    tokens = text.split()

    # Step 6 & 7: Remove stopwords (standard + financial)
    tokens = [t for t in tokens if t not in ALL_STOPWORDS]

    # Step 8 & 9: Lemmatize and filter short tokens
    if USE_SPACY and nlp is not None:
        # Process as a Doc for lemmatization
        doc = nlp(" ".join(tokens))
        tokens = [
            token.lemma_
            for token in doc
            if len(token.lemma_) >= 3
            and token.lemma_ not in ALL_STOPWORDS
            and token.lemma_.isalpha()
        ]
    else:
        # Fallback: simple suffix stripping
        def simple_lemma(word):
            if word.endswith('ing') and len(word) > 5:
                return word[:-3]
            if word.endswith('tion') and len(word) > 6:
                return word[:-4]
            if word.endswith('ed') and len(word) > 4:
                return word[:-2]
            if word.endswith('ies') and len(word) > 4:
                return word[:-3] + 'y'
            if word.endswith('es') and len(word) > 3:
                return word[:-2]
            if word.endswith('s') and len(word) > 3:
                return word[:-1]
            return word

        tokens = [
            simple_lemma(t)
            for t in tokens
            if len(t) >= 3 and t.isalpha()
            and simple_lemma(t) not in ALL_STOPWORDS
        ]

    return " ".join(tokens), tokens

# ── 5. Load corpus ────────────────────────────────────────────
log(f"\nLoading corpus from: {INPUT_PATH.name}")
if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_PATH}\n"
        "Run 01b_cfpb_filter.py and 01g_hud_crosswalk.py first."
    )

df = pd.read_parquet(INPUT_PATH)
log(f"  Rows loaded: {len(df):,}")
log(f"  Columns: {list(df.columns)}")

# ── 6. Identify narrative column ──────────────────────────────
narrative_col = next(
    (c for c in df.columns
     if 'narrative' in c.lower() or 'complaint_text' in c.lower()),
    None
)
if narrative_col is None:
    raise ValueError(
        f"Narrative column not found. Columns: {list(df.columns)}"
    )
log(f"  Narrative column: '{narrative_col}'")

# ── 7. Filter to non-empty narratives ────────────────────────
df = df[df[narrative_col].notna()].copy()
df = df[df[narrative_col].str.strip().str.len() > 0].copy()
log(f"  Rows with non-empty narrative: {len(df):,}")

# ── 8. Add temporal era flag ──────────────────────────────────
if 'year' in df.columns:
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['post_2020'] = (df['year'] >= 2020).astype(int)
    df['era'] = df['post_2020'].map({0: 'pre_2020', 1: 'post_2020'})
    log(f"\n  Pre-2020 complaints:  {(df['post_2020']==0).sum():,}")
    log(f"  Post-2020 complaints: {(df['post_2020']==1).sum():,}")
else:
    df['post_2020'] = np.nan
    df['era'] = 'unknown'
    log("  WARNING: 'year' column not found — era flag not set")

# ── 9. Preprocess all narratives ─────────────────────────────
log(f"\nPreprocessing {len(df):,} narratives...")
log("  This may take 5-15 minutes depending on machine speed...")

from tqdm import tqdm
tqdm.pandas(desc="  Preprocessing")

try:
    results = df[narrative_col].progress_apply(preprocess_narrative)
except ImportError:
    log("  (tqdm not available — running without progress bar)")
    results = df[narrative_col].apply(preprocess_narrative)

df['tokens_joined'] = [r[0] for r in results]
df['token_list']    = [r[1] for r in results]
df['token_count']   = df['token_list'].apply(len)

# ── 10. Quality checks ────────────────────────────────────────
log("\nQuality checks:")
log(f"  Mean tokens per document: {df['token_count'].mean():.1f}")
log(f"  Median tokens:            {df['token_count'].median():.1f}")
log(f"  Min tokens:               {df['token_count'].min()}")
log(f"  Max tokens:               {df['token_count'].max()}")

# Documents with very few tokens (< 10) are flagged
short_docs = (df['token_count'] < 10).sum()
log(f"  Documents with < 10 tokens: {short_docs:,} ({short_docs/len(df)*100:.1f}%)")
df['sufficient_tokens'] = (df['token_count'] >= 10).astype(int)

# Sample 3 preprocessed narratives for manual inspection
log("\nSample preprocessed narratives (first 3 with >= 20 tokens):")
sample = df[df['token_count'] >= 20].head(3)
for i, (_, row) in enumerate(sample.iterrows()):
    log(f"\n  [{i+1}] Original (first 100 chars):")
    log(f"       {str(row[narrative_col])[:100]}...")
    log(f"  [{i+1}] Preprocessed ({row['token_count']} tokens):")
    log(f"       {row['tokens_joined'][:100]}...")

# ── 11. Save output ───────────────────────────────────────────
# Drop token_list (list of lists doesn't serialize to parquet cleanly)
# Keep tokens_joined (string) for all downstream processing
save_cols = [c for c in df.columns if c != 'token_list']
df_save = df[save_cols].copy()

df_save.to_parquet(OUTPUT_PATH, index=False, engine='pyarrow')
log(f"\nSaved: {OUTPUT_PATH}")
log(f"Size:  {OUTPUT_PATH.stat().st_size / 1024 / 1024:.1f} MB")
log(f"Shape: {df_save.shape[0]:,} rows x {df_save.shape[1]} columns")

# ── 12. Write log ─────────────────────────────────────────────
with open(LOG_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))
log(f"Log:   {LOG_PATH}")

log("\n" + "=" * 60)
log("Step 02a complete. Ready for Tracks A, B, C.")
log("=" * 60)
