"""
============================================================
Phase 2 NLP Pipeline — Track A: AI Complaint Density
============================================================
Project: Wisconsin Retirement AI
Script:  code/02_nlp_pipeline/02b_track_a_ai_density.py
Author:  Khawaja Sazzad Ali
Date:    Summer 2026

PURPOSE:
    Classify each CFPB complaint narrative as AI-related or not,
    then aggregate to county-year AI Financial Tool Complaint
    Density per 10,000 residents.

    This is the study's PRIMARY AI EXPOSURE VARIABLE — the
    endogenous regressor instrumented by FCC broadband penetration
    in the IV-Probit specifications.

TWO CLASSIFICATION METHODS:
    Method 1 — Keyword classification with temporally stratified
               dictionaries (pre-2020 vs post-2020 keywords)
    Method 2 — Semantic sentence embeddings using all-MiniLM-L6-v2
               (if sentence-transformers installed)

    Both methods evaluated against 300-narrative validation sample.
    Higher F1 used as primary classifier.
    Agreement rate between methods reported as construct validity.

INPUT:
    data/processed/cfpb_wisconsin_preprocessed.parquet
    data/processed/acs_county_controls.parquet (for population denominator)

OUTPUT:
    data/processed/cfpb_ai_density_county_year.parquet
    code/02_nlp_pipeline/validation/ai_classifier_eval.csv
    docs/track_a_log.txt
============================================================
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ── 0. Paths ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
DOCS_DIR     = PROJECT_ROOT / "docs"
VAL_DIR      = PROJECT_ROOT / "code" / "02_nlp_pipeline" / "validation"

VAL_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH  = PROC_DIR / "cfpb_wisconsin_preprocessed.parquet"
ACS_PATH    = PROC_DIR / "acs_county_controls.parquet"
OUTPUT_PATH = PROC_DIR / "cfpb_ai_density_county_year.parquet"
LOG_PATH    = DOCS_DIR / "track_a_log.txt"

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

log("=" * 60)
log("Track A: AI Financial Tool Complaint Density")
log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 60)

# ── 1. Keyword dictionaries ───────────────────────────────────
# Temporally stratified to reflect how AI tools evolved over time.
# Pre-2020: robo-advisors, automated investing platforms
# Post-2020: LLM-era tools (ChatGPT, AI advisors, etc.)

# Each term uses whole-word matching to avoid false positives
# e.g., 'algorithm' matches 'algorithmic' but not 'algorithmically'

PRE_2020_KEYWORDS = {
    # Robo-advisor platforms (named)
    'wealthfront', 'betterment', 'ellevest', 'acorns',
    'wealthsimple', 'schwab intelligent', 'vanguard digital',
    'fidelity go', 'ally invest',
    # Generic pre-LLM AI finance terms
    'robo advisor', 'robo-advisor', 'roboadvisor',
    'automated advisor', 'automated financial',
    'automated investing', 'automated investment',
    'algorithmic advice', 'algorithmic investing',
    'automated portfolio', 'automated rebalancing',
    'digital advisor', 'digital financial advisor',
    'automated wealth', 'automated asset',
    'smart beta', 'automated trading',
    'digital investment platform',
}

POST_2020_KEYWORDS = {
    # LLM-era tools (generic)
    'chatgpt', 'gpt-4', 'gpt4', 'openai',
    'artificial intelligence', 'ai advisor', 'ai financial',
    'ai investment', 'ai retirement', 'ai planning',
    'ai powered', 'ai-powered', 'ai driven', 'ai-driven',
    'generative ai', 'large language model',
    'chatbot financial', 'chatbot advisor', 'chatbot investment',
    'virtual financial assistant', 'ai assistant financial',
    'automated financial advice', 'automated financial planning',
    'digital financial assistant', 'intelligent advisor',
    'machine learning financial', 'algorithm financial advice',
    # Named post-2020 AI finance tools
    'betterment ai', 'schwab ai', 'fidelity ai',
    'ai wealth management', 'ai portfolio',
    # Also retain robo-advisor terms (still used post-2020)
    'robo advisor', 'robo-advisor', 'roboadvisor',
    'automated advisor', 'automated financial advisor',
    'digital advisor',
}

# Combined for any-era matching
ALL_KEYWORDS = PRE_2020_KEYWORDS | POST_2020_KEYWORDS

log(f"\nKeyword dictionary sizes:")
log(f"  Pre-2020:  {len(PRE_2020_KEYWORDS)} terms")
log(f"  Post-2020: {len(POST_2020_KEYWORDS)} terms")
log(f"  Combined:  {len(ALL_KEYWORDS)} unique terms")


# ── 2. Keyword classifier (Method 1) ─────────────────────────
def classify_keyword(text: str, post_2020: int) -> int:
    """
    Flag narrative as AI-related using era-appropriate keywords.
    Returns 1 if AI-related, 0 otherwise.

    Uses whole-word regex matching with word boundaries.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 0

    # Select keyword set based on era
    keywords = POST_2020_KEYWORDS if post_2020 else PRE_2020_KEYWORDS

    # Also always check the shared terms (robo-advisor era terms
    # still appear in post-2020 complaints)
    keywords = keywords | PRE_2020_KEYWORDS

    text_lower = text.lower()
    for kw in keywords:
        # Whole-phrase matching (handles multi-word terms)
        pattern = r'\b' + re.escape(kw) + r'\b'
        if re.search(pattern, text_lower):
            return 1
    return 0


def get_matched_keywords(text: str, post_2020: int) -> list:
    """Return list of matched keywords for inspection."""
    if not isinstance(text, str):
        return []
    keywords = ALL_KEYWORDS
    text_lower = text.lower()
    matched = []
    for kw in keywords:
        pattern = r'\b' + re.escape(kw) + r'\b'
        if re.search(pattern, text_lower):
            matched.append(kw)
    return matched


# ── 3. Load preprocessed corpus ──────────────────────────────
log(f"\nLoading preprocessed corpus...")
if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Preprocessed file not found: {INPUT_PATH}\n"
        "Run 02a_preprocessing.py first."
    )

df = pd.read_parquet(INPUT_PATH)
log(f"  Rows: {len(df):,}")

# Identify the narrative column for keyword matching
# We apply keywords to ORIGINAL narrative, not preprocessed tokens
# (preprocessing may remove important compound phrases)
narrative_col = next(
    (c for c in df.columns
     if 'narrative' in c.lower() and 'consumer' in c.lower()),
    None
)
if narrative_col is None:
    narrative_col = next(
        (c for c in df.columns if 'narrative' in c.lower()),
        None
    )
if narrative_col is None:
    raise ValueError(f"Narrative column not found. Columns: {list(df.columns)}")

log(f"  Using narrative column: '{narrative_col}'")

# Ensure post_2020 flag exists
if 'post_2020' not in df.columns:
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['post_2020'] = (df['year'] >= 2020).astype(int)
    else:
        df['post_2020'] = 0
        log("  WARNING: No year column — defaulting all to pre-2020 keywords")


# ── 4. Apply Method 1: Keyword classifier ────────────────────
log("\n[Method 1] Applying keyword classifier...")

df['ai_keyword'] = df.apply(
    lambda row: classify_keyword(
        str(row[narrative_col]),
        int(row['post_2020']) if pd.notna(row['post_2020']) else 0
    ),
    axis=1
)

df['matched_keywords'] = df.apply(
    lambda row: get_matched_keywords(
        str(row[narrative_col]),
        int(row['post_2020']) if pd.notna(row['post_2020']) else 0
    ),
    axis=1
)

n_ai_keyword = df['ai_keyword'].sum()
log(f"  AI-flagged by keywords: {n_ai_keyword:,} of {len(df):,} "
    f"({n_ai_keyword/len(df)*100:.2f}%)")

# Show most common matched keywords
all_matched = []
for kw_list in df[df['ai_keyword']==1]['matched_keywords']:
    all_matched.extend(kw_list)

if all_matched:
    from collections import Counter
    kw_counts = Counter(all_matched).most_common(20)
    log("\n  Most common matched keywords:")
    for kw, cnt in kw_counts:
        log(f"    {cnt:>4}  {kw}")


# ── 5. Apply Method 2: Semantic embeddings (if available) ─────
USE_SEMANTIC = False
log("\n[Method 2] Attempting semantic classification...")

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    log("  Loading sentence transformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Seed sentences describing AI financial tool experiences
    SEED_SENTENCES = [
        "I used a robo-advisor to manage my investment portfolio automatically.",
        "The chatbot gave me financial advice about my retirement savings.",
        "An AI tool recommended investment allocations for my 401k.",
        "The automated financial advisor made trades without my approval.",
        "I received financial planning advice from an artificial intelligence system.",
        "The robo-advisor charged fees I did not understand or authorize.",
        "ChatGPT helped me understand my retirement plan options.",
        "The digital investment platform automatically rebalanced my portfolio.",
        "An algorithm made decisions about my financial accounts.",
        "The AI assistant gave me incorrect advice about my investments.",
    ]

    log(f"  Encoding {len(SEED_SENTENCES)} seed sentences...")
    seed_embeddings = model.encode(SEED_SENTENCES, show_progress_bar=False)
    seed_centroid = seed_embeddings.mean(axis=0, keepdims=True)

    # Encode all narratives in batches for efficiency
    log(f"  Encoding {len(df):,} narratives in batches...")
    BATCH_SIZE = 256
    THRESHOLD  = 0.35  # Cosine similarity threshold (calibrated empirically)

    all_scores = []
    narratives = df[narrative_col].fillna('').tolist()

    try:
        from tqdm import tqdm
        batches = range(0, len(narratives), BATCH_SIZE)
        for i in tqdm(batches, desc="  Encoding"):
            batch = narratives[i:i+BATCH_SIZE]
            embeddings = model.encode(batch, show_progress_bar=False)
            scores = cosine_similarity(embeddings, seed_centroid).flatten()
            all_scores.extend(scores.tolist())
    except ImportError:
        for i in range(0, len(narratives), BATCH_SIZE):
            batch = narratives[i:i+BATCH_SIZE]
            embeddings = model.encode(batch, show_progress_bar=False)
            scores = cosine_similarity(embeddings, seed_centroid).flatten()
            all_scores.extend(scores.tolist())
            if i % 2000 == 0:
                log(f"    Processed {i}/{len(narratives)}...")

    df['ai_semantic_score'] = all_scores
    df['ai_semantic'] = (df['ai_semantic_score'] >= THRESHOLD).astype(int)

    n_ai_semantic = df['ai_semantic'].sum()
    log(f"  AI-flagged by semantics: {n_ai_semantic:,} of {len(df):,} "
        f"({n_ai_semantic/len(df)*100:.2f}%)")

    # Agreement between methods
    agree = (df['ai_keyword'] == df['ai_semantic']).mean()
    log(f"  Agreement rate (keyword vs semantic): {agree*100:.1f}%")

    # Use Method 1 (keyword) as primary, Method 2 as robustness
    df['ai_flag_primary']    = df['ai_keyword']
    df['ai_flag_robustness'] = df['ai_semantic']
    USE_SEMANTIC = True
    log("  Both methods available. Keyword = primary, Semantic = robustness.")

except ImportError:
    log("  sentence-transformers not installed.")
    log("  Using keyword classifier as sole method.")
    log("  To enable semantic method: pip install sentence-transformers")
    df['ai_semantic_score'] = np.nan
    df['ai_semantic']       = np.nan
    df['ai_flag_primary']   = df['ai_keyword']
    df['ai_flag_robustness'] = np.nan


# ── 6. Create validation sample ──────────────────────────────
log("\nCreating 300-narrative validation sample for manual labeling...")

# Stratified sample: 150 flagged + 150 not flagged by primary classifier
flagged     = df[df['ai_flag_primary'] == 1]
not_flagged = df[df['ai_flag_primary'] == 0]

n_flag_sample = min(150, len(flagged))
n_noflag_sample = min(150, len(not_flagged))

val_flagged     = flagged.sample(n=n_flag_sample, random_state=42)
val_not_flagged = not_flagged.sample(n=n_noflag_sample, random_state=42)
val_sample      = pd.concat([val_flagged, val_not_flagged], ignore_index=True)
val_sample      = val_sample.sample(frac=1, random_state=42)  # Shuffle

# Save validation sample for manual labeling
val_cols = ['COMPLAINT_ID' if 'COMPLAINT_ID' in df.columns else df.columns[0],
            narrative_col, 'ai_flag_primary', 'year', 'post_2020']
val_cols = [c for c in val_cols if c in df.columns]

val_save = val_sample[val_cols].copy()
val_save['manual_label'] = ''   # Column for researcher to fill in
val_save['notes']        = ''   # Column for researcher notes

val_path = VAL_DIR / "ai_validation_sample_300.csv"
val_save.to_csv(val_path, index=False, encoding='utf-8')
log(f"  Validation sample saved: {val_path}")
log(f"  INSTRUCTIONS: Open this CSV and add 1/0 labels in 'manual_label' column.")
log(f"  Label 1 = complaint is about AI financial tools, 0 = it is not.")
log(f"  Return to run validation evaluation in 02b_validate_classifier.py")


# ── 7. Aggregate to county-year ───────────────────────────────
log("\nAggregating to county-year level...")

# Join updated county FIPS from the fixed cfpb_wisconsin_with_county.parquet
log("  Loading fixed county FIPS from cfpb_wisconsin_with_county.parquet...")
FIXED_CFPB_PATH = PROC_DIR / "cfpb_wisconsin_with_county.parquet"

if FIXED_CFPB_PATH.exists():
    df_fixed = pd.read_parquet(FIXED_CFPB_PATH)
    id_col_fixed = next((c for c in df_fixed.columns if 'complaint_id' in c.lower()), None)
    id_col_pre   = next((c for c in df.columns if 'complaint_id' in c.lower()), None)
    if id_col_fixed and id_col_pre:
        df_fixed[id_col_fixed] = df_fixed[id_col_fixed].astype(str)
        df[id_col_pre]         = df[id_col_pre].astype(str)
        df = df.drop(columns=["primary_county_fips"], errors="ignore")
        df = df.merge(
            df_fixed[[id_col_fixed, "primary_county_fips"]].rename(
                columns={id_col_fixed: id_col_pre}),
            on=id_col_pre, how="left"
        )
        log(f"  County FIPS joined: {df['primary_county_fips'].notna().sum():,} matched")
    else:
        log(f"  complaint_id not found (fixed={id_col_fixed}, pre={id_col_pre})")
else:
    log(f"  Fixed CFPB file not found: {FIXED_CFPB_PATH}")

year_col = "year" if "year" in df.columns else None

# Drop rows without valid 5-digit county FIPS
df_wi = df[df["primary_county_fips"].notna()].copy()
df_wi["county_fips"] = df_wi["primary_county_fips"].astype(str).str.strip().str.zfill(5)
df_wi = df_wi[df_wi["county_fips"].str.match(r"^\d{5}$")].copy()
log(f"  Rows with valid county FIPS: {len(df_wi):,} of {len(df):,}")
log(f"  Unique counties: {df_wi['county_fips'].nunique()}")

if year_col and len(df_wi) > 0:
    df_wi["year_num"] = pd.to_numeric(df_wi[year_col], errors="coerce")

    agg = df_wi.groupby(["county_fips", "year_num"]).agg(
        total_complaints       = (narrative_col, "count"),
        ai_complaints_keyword  = ("ai_flag_primary", "sum"),
        ai_complaints_semantic = ("ai_semantic",
                                  lambda x: x.sum() if x.notna().any() else np.nan),
    ).reset_index().rename(columns={"year_num": "year"})

    # Load county population from ACS for rate denominator
    if ACS_PATH.exists():
        df_acs = pd.read_parquet(ACS_PATH)
        df_acs = df_acs[["county_fips", "acs_year", "pop_total"]].copy()
        df_acs["county_fips"] = df_acs["county_fips"].astype(str)

        def nearest_acs_year(yr):
            acs_years = [2009, 2012, 2015, 2018, 2021, 2022]
            return min(acs_years, key=lambda y: abs(y - yr))

        agg["acs_year"]    = agg["year"].apply(
            lambda y: nearest_acs_year(int(y)) if pd.notna(y) else np.nan)
        agg["county_fips"] = agg["county_fips"].astype(str)
        agg = agg.merge(
            df_acs[["county_fips", "acs_year", "pop_total"]],
            on=["county_fips", "acs_year"], how="left"
        )
        log(f"  Population matched: {agg['pop_total'].notna().sum()} of {len(agg)}")
    else:
        agg["pop_total"] = np.nan
        log("  ACS not found — skipping population denominator")

    # AI complaint density per 10,000 residents
    agg["ai_complaint_density"] = np.where(
        agg["pop_total"].notna() & (agg["pop_total"] > 0),
        agg["ai_complaints_keyword"] / agg["pop_total"] * 10000,
        np.nan
    )
    agg["ai_complaint_density_log"] = np.log1p(agg["ai_complaint_density"])

    # Rolling 3-year average for sparse county-years (< 10 complaints)
    agg = agg.sort_values(["county_fips", "year"]).reset_index(drop=True)
    sparse_mask = agg["total_complaints"] < 10
    log(f"  Sparse county-years (< 10 complaints): {sparse_mask.sum()}")

    agg["density_rolling3"] = (
        agg.groupby("county_fips")["ai_complaint_density"]
        .transform(lambda x: x.rolling(window=3, min_periods=1, center=True).mean())
    )
    agg["ai_complaint_density_final"] = np.where(
        sparse_mask, agg["density_rolling3"], agg["ai_complaint_density"]
    )
    agg["ai_complaint_density_final_log"] = np.log1p(agg["ai_complaint_density_final"])

    log(f"\nCounty-year aggregation complete:")
    log(f"  Total county-year cells: {len(agg):,}")
    log(f"  Counties: {agg['county_fips'].nunique()}")
    log(f"  Years: {sorted(agg['year'].dropna().unique().tolist())}")
    log(f"  AI complaints total: {agg['ai_complaints_keyword'].sum():,.0f}")
    log(f"  Mean AI density: {agg['ai_complaint_density_final'].mean():.4f} per 10,000")

    log("\n  AI complaints by year (Wisconsin):")
    yr_summary = agg.groupby("year").agg(
        counties     = ("county_fips", "nunique"),
        total_compl  = ("total_complaints", "sum"),
        ai_compl     = ("ai_complaints_keyword", "sum"),
        mean_density = ("ai_complaint_density_final", "mean")
    ).reset_index()
    for _, row in yr_summary.iterrows():
        pct = row["ai_compl"]/row["total_compl"]*100 if row["total_compl"] > 0 else 0
        log(f"    {int(row['year'])}: {int(row['total_compl']):>5} total | "
            f"{int(row['ai_compl']):>4} AI ({pct:.1f}%) | "
            f"density={row['mean_density']:.4f}")

    agg.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow")
    log(f"\nSaved: {OUTPUT_PATH}")
else:
    log("  Cannot aggregate — no rows with valid county FIPS and year.")


# ── 8. Write log ─────────────────────────────────────────────
with open(LOG_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))
log(f"Log: {LOG_PATH}")

log("\n" + "=" * 60)
log("Track A complete.")
log(f"Next: Open validation/ai_validation_sample_300.csv")
log(f"      Label 300 narratives manually (1=AI related, 0=not)")
log(f"      Then run 02e_validate_classifier.py for F1 scoring")
log("=" * 60)
