"""
============================================================
Track B: LDA Topic Modeling on CFPB Complaint Narratives
============================================================
Project: Wisconsin Retirement AI
Script:  code/02_nlp_pipeline/02c_track_b_topic_model.py
Author:  Khawaja Sazzad Ali
Date:    Summer 2026

PURPOSE:
    Identify latent behavioral themes in Wisconsin CFPB complaint
    narratives to validate that the behavioral constructs in the
    FINRA NFCS survey correspond to documented consumer experiences.

    Expected topics include:
      - Enrollment confusion
      - Penalty misunderstanding
      - Automatic deduction disputes
      - Account access issues
      - Investment product confusion

    This is a CONSTRUCT VALIDITY CHECK, not a regression variable.
    Topic prevalence by county and year is saved for descriptive
    analysis but does not enter the main Probit models.

LDA IMPLEMENTATION:
    Primary:  sklearn LatentDirichletAllocation (no C++ compiler needed,
              works on all platforms including Windows)
    Optional: gensim LDA (if installed via conda — produces identical
              results with better coherence scoring tools)

INPUT:   data/processed/cfpb_wisconsin_preprocessed.parquet
         (produced by 02a_preprocessing.py)
OUTPUT:  data/processed/cfpb_topic_prevalence_county_year.parquet
         outputs/figures/lda_topic_terms.png
         docs/lda_topic_labels.txt
============================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ── 0. Paths ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
DOCS_DIR     = PROJECT_ROOT / "docs"
FIG_DIR      = PROJECT_ROOT / "outputs" / "figures"

PROC_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Check for gensim (optional) ───────────────────────────
try:
    import gensim
    from gensim import corpora
    from gensim.models import LdaModel
    from gensim.models.coherencemodel import CoherenceModel
    USE_GENSIM = True
    print("gensim available — using gensim LDA with coherence scoring")
except ImportError:
    USE_GENSIM = False
    print("gensim not installed — using sklearn LDA (equivalent results)")
    print("To install gensim: conda install -c conda-forge gensim")

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

print("=" * 60)
print("Track B: LDA Topic Modeling")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Backend: {'gensim' if USE_GENSIM else 'sklearn'}")
print("=" * 60)

# ── 2. Load preprocessed corpus ──────────────────────────────
input_path = PROC_DIR / "cfpb_wisconsin_preprocessed.parquet"
if not input_path.exists():
    raise FileNotFoundError(
        f"Preprocessed file not found: {input_path}\n"
        "Run 02a_preprocessing.py first."
    )

df = pd.read_parquet(input_path)
print(f"\nLoaded: {len(df):,} documents")

# Use the cleaned token string for topic modeling
text_col = 'tokens_joined'  # space-joined lemmatized tokens from preprocessing
if text_col not in df.columns:
    # Fallback: use original narrative if preprocessing output differs
    text_col = next(
        (c for c in df.columns if 'token' in c or 'clean' in c or 'text' in c),
        None
    )
    if text_col is None:
        raise ValueError(f"Cannot find text column. Columns: {list(df.columns)}")

print(f"Text column: '{text_col}'")

# Filter to documents with sufficient tokens
docs = df[text_col].fillna('').astype(str)
doc_lengths = docs.str.split().str.len()
min_tokens = 10
mask = doc_lengths >= min_tokens
docs_filtered = docs[mask]
df_filtered = df[mask].copy()

print(f"Documents with >= {min_tokens} tokens: {mask.sum():,} of {len(df):,}")

# ── 3. Select number of topics K ─────────────────────────────
# Based on prior CFPB LDA literature, K=10-15 is typical.
# We test K=8,10,12,15 and select by perplexity (sklearn)
# or coherence score (gensim).

K_RANGE    = [8, 10, 12, 15]
K_FINAL    = 12    # Default; overridden by search below
MAX_ITER   = 20    # LDA iterations (increase to 50 for final run)
RANDOM_STATE = 42

print(f"\nTesting K values: {K_RANGE}")
print(f"Max iterations per model: {MAX_ITER}")

# ── 4. Vectorize text ─────────────────────────────────────────
print("\nVectorizing documents...")

vectorizer = CountVectorizer(
    max_features = 5000,    # Top 5000 terms by frequency
    min_df       = 5,       # Term must appear in at least 5 docs
    max_df       = 0.95,    # Exclude terms in >95% of docs
    ngram_range  = (1, 2),  # Unigrams and bigrams
    token_pattern = r'\b[a-zA-Z][a-zA-Z]+\b'  # Words only, min 2 chars
)

dtm = vectorizer.fit_transform(docs_filtered)
vocab = vectorizer.get_feature_names_out()
print(f"Vocabulary size: {len(vocab):,} terms")
print(f"Document-term matrix: {dtm.shape}")

# ── 5. Fit LDA models and select K ───────────────────────────
if USE_GENSIM:
    # ── Gensim path ───────────────────────────────────────────
    print("\nFitting gensim LDA models...")

    # Convert sklearn DTM to gensim corpus format
    corpus_gensim = gensim.matutils.Sparse2Corpus(dtm, documents_columns=False)
    id2word = {i: term for i, term in enumerate(vocab)}

    best_coherence = -np.inf
    best_k = K_FINAL
    coherence_scores = {}

    for k in K_RANGE:
        print(f"  K={k}...", end=" ")
        model = LdaModel(
            corpus      = corpus_gensim,
            id2word     = id2word,
            num_topics  = k,
            passes      = MAX_ITER,
            random_state = RANDOM_STATE,
            alpha       = 'auto',
            per_word_topics = False
        )
        # Coherence score (C_v)
        coh = CoherenceModel(
            model   = model,
            texts   = [t.split() for t in docs_filtered],
            dictionary = corpora.Dictionary.from_corpus(
                corpus_gensim, id2word=id2word
            ),
            coherence = 'c_v'
        )
        score = coh.get_coherence()
        coherence_scores[k] = score
        print(f"coherence={score:.4f}")
        if score > best_coherence:
            best_coherence = score
            best_k = k
            best_model = model

    print(f"\nBest K: {best_k} (coherence={best_coherence:.4f})")

    # Get topic-term distributions
    def get_topic_terms(model, n=15):
        topics = {}
        for i in range(model.num_topics):
            terms = model.show_topic(i, topn=n)
            topics[i] = [(term, round(prob, 4)) for term, prob in terms]
        return topics

    topic_terms = get_topic_terms(best_model)

    # Get document-topic distributions
    doc_topics = []
    for doc_bow in corpus_gensim:
        topic_dist = dict(best_model.get_document_topics(doc_bow, minimum_probability=0))
        doc_topics.append([topic_dist.get(i, 0.0) for i in range(best_k)])

    doc_topic_df = pd.DataFrame(
        doc_topics,
        columns=[f"topic_{i}" for i in range(best_k)]
    )

else:
    # ── sklearn path (default on Windows) ────────────────────
    print("\nFitting sklearn LDA models (this may take a few minutes)...")

    best_perplexity = np.inf
    best_k = K_FINAL
    perplexity_scores = {}

    for k in K_RANGE:
        print(f"  K={k}...", end=" ", flush=True)
        lda = LatentDirichletAllocation(
            n_components     = k,
            max_iter         = MAX_ITER,
            learning_method  = 'online',   # Faster than 'batch' for large corpora
            random_state     = RANDOM_STATE,
            n_jobs           = -1          # Use all CPU cores
        )
        lda.fit(dtm)
        perp = lda.perplexity(dtm)
        perplexity_scores[k] = perp
        print(f"perplexity={perp:.1f}")

        # Lower perplexity = better fit
        if perp < best_perplexity:
            best_perplexity = perp
            best_k = k
            best_model = lda

    print(f"\nBest K: {best_k} (perplexity={best_perplexity:.1f})")
    print("Note: perplexity decreases with K — also inspect topic interpretability")

    # Get top terms per topic
    def get_top_terms_sklearn(model, vocab, n=15):
        topics = {}
        for i, topic_vec in enumerate(model.components_):
            top_idx = topic_vec.argsort()[-n:][::-1]
            topics[i] = [(vocab[j], round(topic_vec[j] / topic_vec.sum(), 4))
                         for j in top_idx]
        return topics

    topic_terms = get_top_terms_sklearn(best_model, vocab)

    # Document-topic matrix
    doc_topic_arr = best_model.transform(dtm)
    doc_topic_df = pd.DataFrame(
        doc_topic_arr,
        columns=[f"topic_{i}" for i in range(best_k)]
    )

# ── 6. Print and label topics ─────────────────────────────────
print("\n" + "=" * 60)
print("TOPIC TERMS (top 15 per topic)")
print("=" * 60)

# Expected topic labels for CFPB retirement-related complaints
EXPECTED_LABELS = {
    "enroll":    "Enrollment Confusion",
    "penalty":   "Early Withdrawal Penalty",
    "withdraw":  "Hardship Withdrawal",
    "401k":      "401k/Employer Plan Issues",
    "ira":       "IRA Account Issues",
    "deduct":    "Automatic Deduction Disputes",
    "fee":       "Fee Disputes",
    "invest":    "Investment Product Confusion",
    "access":    "Account Access Issues",
    "transfer":  "Fund Transfer Issues",
    "chatbot":   "AI/Automated Tool Issues",
    "robo":      "Robo-Advisor Issues",
}

topic_labels = {}
log_lines = ["LDA Topic Labels\n", f"Generated: {datetime.now()}\n",
             f"Backend: {'gensim' if USE_GENSIM else 'sklearn'}\n",
             f"K={best_k}\n\n"]

for topic_id, terms in topic_terms.items():
    term_words = [t[0] for t in terms]
    term_str   = ", ".join(term_words[:10])

    # Auto-assign label based on top terms
    label = f"Topic {topic_id}"
    for keyword, candidate_label in EXPECTED_LABELS.items():
        if any(keyword in w for w in term_words[:8]):
            label = candidate_label
            break

    topic_labels[topic_id] = label
    print(f"\nTopic {topic_id:2d} [{label}]")
    print(f"  Terms: {term_str}")

    log_lines.append(f"Topic {topic_id}: {label}\n")
    log_lines.append(f"  Terms: {term_str}\n\n")

# ── 7. Aggregate to county-year level ────────────────────────
print("\nAggregating topic prevalence to county-year level...")

# Merge document-topic distributions back to metadata
doc_topic_df.index = df_filtered.index
df_with_topics = df_filtered.join(doc_topic_df)

# Identify county and year columns
county_col = next(
    (c for c in df_with_topics.columns if 'county' in c.lower()),
    None
)
year_col = 'year' if 'year' in df_with_topics.columns else None

if county_col and year_col:
    topic_cols = [f"topic_{i}" for i in range(best_k)]
    county_year_topics = (
        df_with_topics
        .groupby([county_col, year_col])[topic_cols]
        .mean()
        .reset_index()
        .rename(columns={county_col: 'county_fips', year_col: 'year'})
    )

    # Rename columns to include labels
    rename_map = {
        f"topic_{i}": f"topic_{i}_{topic_labels.get(i, f'topic{i}').replace(' ', '_').lower()}"
        for i in range(best_k)
    }
    county_year_topics = county_year_topics.rename(columns=rename_map)

    out_path = PROC_DIR / "cfpb_topic_prevalence_county_year.parquet"
    county_year_topics.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")
    print(f"Shape: {county_year_topics.shape}")
else:
    print(f"WARNING: county_col={county_col}, year_col={year_col}")
    print("Topic prevalence not aggregated — county/year columns not found.")
    print("This script must run after 01g_hud_crosswalk.py adds county FIPS.")

# ── 8. Save topic label log ───────────────────────────────────
log_path = DOCS_DIR / "lda_topic_labels.txt"
with open(log_path, 'w') as f:
    f.writelines(log_lines)
print(f"Topic labels saved: {log_path}")

# ── 9. Optional: visualize top terms ─────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_show = min(best_k, 6)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(n_show):
        terms_plot = topic_terms[i][:10]
        words  = [t[0] for t in terms_plot]
        scores = [t[1] for t in terms_plot]
        ax = axes[i]
        ax.barh(words[::-1], scores[::-1], color='steelblue')
        ax.set_title(f"Topic {i}: {topic_labels.get(i, '')}", fontsize=10)
        ax.set_xlabel("Weight")

    plt.suptitle(f"LDA Topic Terms — Wisconsin CFPB Complaints (K={best_k})",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig_path = FIG_DIR / "lda_topic_terms.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {fig_path}")

except Exception as e:
    print(f"Figure not saved (matplotlib issue): {e}")

print("\n" + "=" * 60)
print("Track B complete.")
print(f"Backend used: {'gensim' if USE_GENSIM else 'sklearn'}")
print("=" * 60)
