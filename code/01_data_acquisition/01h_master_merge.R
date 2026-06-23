# ============================================================
# Workstream 1H: Master Dataset Construction
# Merges all processed workstreams into single analytical file
# ============================================================
# Project: Wisconsin Retirement AI
# Script:  code/01_data_acquisition/01h_master_merge.R
# Author:  Khawaja Sazzad Ali
# Date:    Summer 2026
#
# PREREQUISITE FILES (all must exist in data/processed/):
#   nfcs_wisconsin_pooled.parquet      <- from 01a_nfcs_process.py
#   cfpb_wisconsin_with_county.parquet <- from 01b + 01g
#   fcc_broadband_county_year.parquet  <- from 01c
#   acs_county_controls.parquet        <- from 01e
#   bls_unemployment_county_year.parquet <- from 01f
#
# NOTE: CFPB NLP outputs (AI density, distress index) are merged
#       separately after Phase 2 NLP pipeline runs.
#       This script merges all non-NLP county controls now.
#
# OUTPUT:
#   data/final/master_dataset.parquet       <- full dataset
#   data/final/master_dataset_pre_nlp.parquet <- without NLP vars
#   docs/merge_log.md
# ============================================================

library(tidyverse)
library(arrow)
library(here)
library(glue)

# ── 0. Paths ─────────────────────────────────────────────────
proc_dir  <- here("data", "processed")
final_dir <- here("data", "final")
docs_dir  <- here("docs")
dir.create(final_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(docs_dir,  showWarnings = FALSE, recursive = TRUE)

# ── 1. Merge log infrastructure ───────────────────────────────
merge_steps <- list()
step_n <- 0

record_merge <- function(label, n_before, n_after, n_unmatched = NA,
                         key = NA) {
  step_n <<- step_n + 1
  cat(sprintf(
    "\nStep %d: %s\n  Before: %s | After: %s | Unmatched: %s\n",
    step_n, label,
    format(n_before, big.mark = ","),
    format(n_after,  big.mark = ","),
    if (is.na(n_unmatched)) "N/A" else format(n_unmatched, big.mark = ",")
  ))
  merge_steps[[step_n]] <<- list(
    step = step_n, label = label,
    n_before = n_before, n_after = n_after,
    n_unmatched = n_unmatched, key = key
  )
}

# ── 2. Helper: load parquet with existence check ──────────────
load_pq <- function(filename) {
  path <- file.path(proc_dir, filename)
  if (!file.exists(path)) {
    warning(glue("Missing: {path}"))
    return(NULL)
  }
  df <- read_parquet(path)
  cat(glue("  Loaded {filename}: {format(nrow(df), big.mark=',')} rows\n\n"))
  df
}

cat(strrep("=", 60), "\n")
cat("Master Dataset Construction\n")
cat(format(Sys.time()), "\n")
cat(strrep("=", 60), "\n\n")

# ── 3. Load all processed files ───────────────────────────────
cat("Loading processed files...\n")
df_nfcs <- load_pq("nfcs_wisconsin_pooled.parquet")
df_fcc  <- load_pq("fcc_broadband_county_year.parquet")
df_acs  <- load_pq("acs_county_controls.parquet")
df_bls  <- load_pq("bls_unemployment_county_year.parquet")

if (is.null(df_nfcs)) stop("NFCS file missing. Run 01a_nfcs_process.py first.")

# ── 4. Start with NFCS backbone ──────────────────────────────
cat(strrep("-", 60), "\n")
cat("MERGE SEQUENCE\n")
cat(strrep("-", 60), "\n")

master <- df_nfcs

# Standardize key columns
master <- master %>%
  mutate(
    SURVEY_WAVE  = as.integer(SURVEY_WAVE),
    county_fips  = as.character(county_fips),
    NFCSID       = as.character(NFCSID)
  )

cat(glue("\nBackbone (NFCS Wisconsin): {format(nrow(master), big.mark=',')} respondents\n"))
cat(glue("Survey waves: {paste(sort(unique(master$SURVEY_WAVE)), collapse=', ')}\n"))
cat(glue("County FIPS present: {sum(!is.na(master$county_fips))}\n\n"))


# ── 5. Merge FCC Broadband (IV for AI exposure) ───────────────
if (!is.null(df_fcc)) {
  n_before <- nrow(master)

  df_fcc_merge <- df_fcc %>%
    mutate(
      county_fips = as.character(county_fips),
      year        = as.integer(year)
    ) %>%
    select(county_fips, year, bb_providers_25_3, bb_providers_100_10,
           era) %>%
    rename(fcc_year = year)

  # Match each NFCS respondent to FCC data for their wave year
  # FCC years available: 2008-2025
  # NFCS waves: 2009, 2012, 2015, 2018, 2021, 2024
  # Direct year match works since FCC covers all those years

  master <- master %>%
    left_join(
      df_fcc_merge %>% rename(SURVEY_WAVE = fcc_year),
      by = c("county_fips", "SURVEY_WAVE")
    )

  n_unmatched <- sum(is.na(master$bb_providers_25_3))
  record_merge("FCC Broadband (IV)", n_before, nrow(master),
               n_unmatched, "county_fips + SURVEY_WAVE")
}


# ── 6. Merge ACS County Controls ─────────────────────────────
if (!is.null(df_acs)) {
  n_before <- nrow(master)

  df_acs_merge <- df_acs %>%
    mutate(
      county_fips = as.character(county_fips),
      nfcs_wave   = as.integer(nfcs_wave)
    ) %>%
    select(county_fips, nfcs_wave,
           pop_total, median_hh_income, log_median_income,
           poverty_rate, pct_bach_plus,
           pct_white_nh, pct_black_nh, pct_hispanic, pct_asian_nh,
           pct_owner_occ, pct_broadband)

  master <- master %>%
    left_join(
      df_acs_merge,
      by = c("county_fips", "SURVEY_WAVE" = "nfcs_wave")
    )

  n_unmatched <- sum(is.na(master$poverty_rate))
  record_merge("ACS County Controls", n_before, nrow(master),
               n_unmatched, "county_fips + SURVEY_WAVE->nfcs_wave")
}


# ── 7. Merge BLS Unemployment ─────────────────────────────────
if (!is.null(df_bls)) {
  n_before <- nrow(master)

  df_bls_merge <- df_bls %>%
    mutate(
      county_fips = as.character(county_fips),
      year        = as.integer(year)
    ) %>%
    select(county_fips, year, unemp_rate)

  master <- master %>%
    left_join(
      df_bls_merge,
      by = c("county_fips", "SURVEY_WAVE" = "year")
    )

  n_unmatched <- sum(is.na(master$unemp_rate))
  record_merge("BLS Unemployment Rate", n_before, nrow(master),
               n_unmatched, "county_fips + SURVEY_WAVE")
}


# ── 8. Variable cleaning and type enforcement ─────────────────
cat("\nCleaning variable types...\n")

# Numeric enforcement for all analytical variables
num_vars <- c(
  # Outcome
  "enrolled",
  # Behavioral
  "literacy_score", "impatience_index",
  "bias_payday", "bias_overdraft", "bias_cc_revolve",
  # Demographics
  "AGE_NUM", "income", "education", "marital",
  "homeowner", "fin_fragility", "ret_calc",
  # Derived indicators
  "minority", "low_income",
  # Financial education
  "fin_ed_received", "fin_ed_school",
  # AI interest (2024 only)
  "ai_interest",
  # County-level IV and controls
  "bb_providers_25_3", "bb_providers_100_10",
  "poverty_rate", "log_median_income", "median_hh_income",
  "pct_bach_plus", "pct_white_nh", "pct_black_nh",
  "pct_hispanic", "pct_asian_nh",
  "pct_owner_occ", "pct_broadband", "pop_total",
  "unemp_rate",
  # Survey weights
  "weight_nat", "weight_st"
)

for (v in num_vars) {
  if (v %in% names(master)) {
    master[[v]] <- suppressWarnings(as.numeric(master[[v]]))
  }
}

# Character enforcement
char_vars <- c("NFCSID", "county_fips", "zip_clean")
for (v in char_vars) {
  if (v %in% names(master)) {
    master[[v]] <- as.character(master[[v]])
  }
}

# Integer enforcement
int_vars <- c("SURVEY_WAVE", "STATEQ_NUM")
for (v in int_vars) {
  if (v %in% names(master)) {
    master[[v]] <- suppressWarnings(as.integer(master[[v]]))
  }
}


# ── 9. Construct interaction terms for regression ─────────────
cat("Constructing interaction terms...\n")

master <- master %>%
  mutate(
    # Binary indicators for heterogeneity models
    female     = if_else(!is.na(weight_nat), NA_real_, NA_real_),
    # NOTE: Gender variable requires checking wave-specific column names
    # A50A (2021/2024), A3B (earlier) -- added to script after column audit
    # Placeholder: will be added in variable construction phase

    # Low literacy flag (bottom 40% of distribution)
    low_literacy = if_else(literacy_score <= 2, 1L, 0L, missing = NA_integer_),

    # Wave fixed effect factor
    wave_fe = factor(SURVEY_WAVE),

    # Log transformation of broadband for regression
    log_bb_25_3 = log(bb_providers_25_3 + 1),

    # Standardized literacy score (mean 0, SD 1) for interpretation
    literacy_std = as.numeric(scale(literacy_score)),

    # Standardized impatience index
    impatience_std = as.numeric(scale(impatience_index))
  )


# ── 10. Missingness report ────────────────────────────────────
cat("\nMissingness report for key regression variables:\n")

key_vars_report <- c(
  "enrolled", "literacy_score", "impatience_index",
  "bb_providers_25_3", "poverty_rate", "unemp_rate",
  "log_median_income", "pct_bach_plus",
  "minority", "low_income", "county_fips", "zip_clean"
)

miss_report <- tibble(
  variable  = key_vars_report,
  n_total   = nrow(master),
  n_missing = sapply(key_vars_report, function(v) {
    if (v %in% names(master)) sum(is.na(master[[v]])) else NA_integer_
  }),
  pct_miss  = round(n_missing / n_total * 100, 1)
) %>%
  filter(!is.na(n_missing))

print(miss_report, n = 30)

# Write missingness report to docs
miss_path <- file.path(docs_dir, "missing_data_report.csv")
write_csv(miss_report, miss_path)
cat(glue("\nMissingness report saved: {miss_path}\n"))


# ── 11. Final dataset summary ─────────────────────────────────
cat("\n", strrep("=", 60), "\n", sep = "")
cat("FINAL MASTER DATASET SUMMARY\n")
cat(strrep("=", 60), "\n")
cat(glue("Observations:     {format(nrow(master), big.mark=',')}\n"))
cat(glue("Variables:        {ncol(master)}\n"))
cat(glue("Survey waves:     {paste(sort(unique(master$SURVEY_WAVE)), collapse=', ')}\n"))
cat(glue("Enrollment rate:  {round(mean(master$enrolled, na.rm=TRUE)*100, 1)}%\n"))
cat(glue("Mean literacy:    {round(mean(master$literacy_score, na.rm=TRUE), 2)}/5\n"))
cat(glue("Mean impatience:  {round(mean(master$impatience_index, na.rm=TRUE), 2)}/3\n"))
cat(glue("County FIPS:      {sum(!is.na(master$county_fips))}/{nrow(master)}\n"))
cat(glue("Broadband IV:     {sum(!is.na(master$bb_providers_25_3))}/{nrow(master)} have data\n"))
cat(glue("ACS controls:     {sum(!is.na(master$poverty_rate))}/{nrow(master)} have data\n"))
cat(glue("BLS unemp:        {sum(!is.na(master$unemp_rate))}/{nrow(master)} have data\n"))

cat("\nBy wave:\n")
master %>%
  group_by(SURVEY_WAVE) %>%
  summarise(
    n          = n(),
    enroll_pct = round(mean(enrolled, na.rm=TRUE)*100, 1),
    literacy   = round(mean(literacy_score, na.rm=TRUE), 2),
    impatience = round(mean(impatience_index, na.rm=TRUE), 2),
    pct_county = round(mean(!is.na(county_fips))*100, 1),
    pct_bb     = round(mean(!is.na(bb_providers_25_3))*100, 1),
    .groups    = "drop"
  ) %>%
  print(n = 10)


# ── 12. Save outputs ──────────────────────────────────────────
# Pre-NLP master (county controls only — ready for descriptive analysis)
pre_nlp_path <- file.path(final_dir, "master_dataset_pre_nlp.parquet")
write_parquet(master, pre_nlp_path)
cat(glue("\nSaved (pre-NLP): {pre_nlp_path}\n"))
cat(glue("Size: {round(file.size(pre_nlp_path)/1024, 1)} KB\n"))

# Also save as the working master (will be overwritten after NLP merge)
master_path <- file.path(final_dir, "master_dataset.parquet")
write_parquet(master, master_path)
cat(glue("Saved (master):  {master_path}\n"))


# ── 13. Write merge log ───────────────────────────────────────
log_lines <- c(
  "# Merge Log — Wisconsin Retirement AI",
  glue("Generated: {format(Sys.time())}"),
  glue("Final dataset: {nrow(master)} rows x {ncol(master)} columns"),
  "",
  "## Merge Steps",
  ""
)

for (s in merge_steps) {
  log_lines <- c(log_lines,
    glue("### Step {s$step}: {s$label}"),
    glue("- Rows before:  {format(s$n_before, big.mark=',')}"),
    glue("- Rows after:   {format(s$n_after,  big.mark=',')}"),
    if (!is.na(s$n_unmatched))
      glue("- Unmatched:    {format(s$n_unmatched, big.mark=',')}"),
    if (!is.na(s$key))
      glue("- Merge key:    {s$key}"),
    ""
  )
}

log_lines <- c(log_lines,
  "## Variables in Master Dataset",
  paste(sort(names(master)), collapse = ", "),
  "",
  "## Next Step",
  "Run Phase 2 NLP pipeline (code/02_nlp_pipeline/run_pipeline.py)",
  "Then run 01h_master_merge_nlp.R to add NLP outputs to master dataset."
)

log_path <- file.path(docs_dir, "merge_log.md")
writeLines(log_lines, log_path)
cat(glue("Merge log saved: {log_path}\n"))

cat("\n", strrep("=", 60), "\n", sep = "")
cat("Workstream 1H complete.\n")
cat("Phase 1 data acquisition is DONE.\n")
cat("Next: Run Phase 2 NLP pipeline.\n")
cat(strrep("=", 60), "\n", sep = "")
