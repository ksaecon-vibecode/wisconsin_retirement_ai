# ============================================================
# NLP Outputs Merge — Add Phase 2 results to master dataset
# ============================================================
# Project: Wisconsin Retirement AI
# Script:  code/01_data_acquisition/01h_master_merge_nlp.R
# Author:  Khawaja Sazzad Ali
# Date:    Summer 2026
#
# PURPOSE:
#   Merge the three NLP outputs from Phase 2 into the master
#   dataset. Run this AFTER run_pipeline.py completes.
#
# INPUTS:
#   data/final/master_dataset_pre_nlp.parquet  <- from 01h
#   data/processed/cfpb_ai_density_county_year.parquet   <- Track A
#   data/processed/cfpb_topic_prevalence_county_year.parquet <- Track B
#   data/processed/cfpb_distress_index_county_year.parquet   <- Track C
#
# OUTPUT:
#   data/final/master_dataset.parquet  <- FINAL analytical file
# ============================================================

library(tidyverse)
library(arrow)
library(here)
library(glue)

proc_dir  <- here("data", "processed")
final_dir <- here("data", "final")
docs_dir  <- here("docs")

cat(strrep("=", 60), "\n")
cat("NLP Outputs Merge\n")
cat(format(Sys.time()), "\n")
cat(strrep("=", 60), "\n\n")

# ── 1. Load pre-NLP master ────────────────────────────────────
pre_nlp_path <- file.path(final_dir, "master_dataset_pre_nlp.parquet")
if (!file.exists(pre_nlp_path)) {
  stop("Pre-NLP master not found. Run 01h_master_merge.R first.")
}

master <- read_parquet(pre_nlp_path)
cat(glue("Pre-NLP master loaded: {format(nrow(master), big.mark=',')} rows\n\n"))

# ── 2. Load NLP outputs ───────────────────────────────────────
load_nlp <- function(filename) {
  path <- file.path(proc_dir, filename)
  if (!file.exists(path)) {
    warning(glue("NLP output not found: {filename}\nRun run_pipeline.py first."))
    return(NULL)
  }
  df <- read_parquet(path)
  df$county_fips <- as.character(df$county_fips)
  df$year        <- as.integer(df$year)
  cat(glue("  Loaded {filename}: {format(nrow(df), big.mark=',')} rows\n"))
  df
}

cat("Loading NLP outputs...\n")
df_ai    <- load_nlp("cfpb_ai_density_county_year.parquet")
df_topic <- load_nlp("cfpb_topic_prevalence_county_year.parquet")
df_dist  <- load_nlp("cfpb_distress_index_county_year.parquet")
cat("\n")

# ── 3. Prepare master for merge ───────────────────────────────
master <- master %>%
  mutate(
    county_fips = as.character(county_fips),
    SURVEY_WAVE = as.integer(SURVEY_WAVE)
  )

# ── 4. Merge Track A — AI Complaint Density ──────────────────
if (!is.null(df_ai)) {
  n_before <- nrow(master)

  ai_cols <- c("county_fips", "year",
                "ai_complaint_density_final",
                "ai_complaint_density_final_log",
                "ai_complaint_density_kw_final",
                "ai_complaint_density_kw_final_log",
                "ai_complaints_keyword",
                "ai_complaints_semantic",
                "total_complaints")
  ai_cols <- ai_cols[ai_cols %in% names(df_ai)]

  master <- master %>%
    left_join(
      df_ai %>% select(all_of(ai_cols)),
      by = c("county_fips", "SURVEY_WAVE" = "year")
    )

  matched <- sum(!is.na(master$ai_complaint_density_final))
  cat(glue("Track A (AI density) merged: {matched}/{nrow(master)} matched\n"))
}

# ── 5. Merge Track C — Financial Distress Index ──────────────
if (!is.null(df_dist)) {
  dist_cols <- c("county_fips", "year",
                  "distress_index",
                  "mean_vader_compound",
                  "pct_highly_negative")
  dist_cols <- dist_cols[dist_cols %in% names(df_dist)]

  master <- master %>%
    left_join(
      df_dist %>% select(all_of(dist_cols)),
      by = c("county_fips", "SURVEY_WAVE" = "year")
    )

  matched <- sum(!is.na(master$distress_index))
  cat(glue("Track C (distress) merged:   {matched}/{nrow(master)} matched\n"))
}

# ── 6. Merge Track B — Topic Prevalence (optional control) ───
if (!is.null(df_topic)) {
  # Topic columns start with 'topic_'
  topic_cols <- c("county_fips", "year",
                   names(df_topic)[grepl("^topic_", names(df_topic))])
  topic_cols <- topic_cols[topic_cols %in% names(df_topic)]

  master <- master %>%
    left_join(
      df_topic %>% select(all_of(topic_cols)),
      by = c("county_fips", "SURVEY_WAVE" = "year")
    )

  cat(glue("Track B (topics) merged:     OK\n"))
}

# ── 7. Final missingness check ────────────────────────────────
cat("\nMissingness — NLP variables:\n")
nlp_vars <- c("ai_complaint_density_final",
               "ai_complaint_density_final_log",
               "distress_index")
for (v in nlp_vars) {
  if (v %in% names(master)) {
    n_miss <- sum(is.na(master[[v]]))
    pct    <- round(n_miss / nrow(master) * 100, 1)
    cat(glue("  {v}: {n_miss} missing ({pct}%)\n"))
  } else {
    cat(glue("  {v}: NOT IN DATASET\n"))
  }
}

# ── 8. Final summary ──────────────────────────────────────────
cat("\n", strrep("=", 60), "\n", sep = "")
cat("FINAL ANALYTICAL DATASET SUMMARY\n")
cat(strrep("=", 60), "\n")
cat(glue("Observations:  {format(nrow(master), big.mark=',')}\n"))
cat(glue("Variables:     {ncol(master)}\n"))
cat(glue("Waves:         {paste(sort(unique(master$SURVEY_WAVE)), collapse=', ')}\n"))
cat(glue("Enrollment:    {round(mean(master$enrolled, na.rm=TRUE)*100,1)}%\n"))
cat(glue("Literacy:      {round(mean(master$literacy_score, na.rm=TRUE),2)}/5\n"))
cat(glue("Impatience:    {round(mean(master$impatience_index, na.rm=TRUE),2)}/3\n"))

if ("ai_complaint_density_final" %in% names(master)) {
  ai_mean <- round(mean(master$ai_complaint_density_final, na.rm=TRUE), 4)
  cat(glue("AI density:    {ai_mean} per 10,000 residents\n"))
}
if ("distress_index" %in% names(master)) {
  dist_mean <- round(mean(master$distress_index, na.rm=TRUE), 4)
  cat(glue("Distress idx:  {dist_mean}\n"))
}

# ── 9. Save final master ──────────────────────────────────────
out_path <- file.path(final_dir, "master_dataset.parquet")
write_parquet(master, out_path)
cat(glue("\nFinal master saved: {out_path}\n"))
cat(glue("Size: {round(file.size(out_path)/1024, 1)} KB\n"))

cat("\n", strrep("=", 60), "\n", sep = "")
cat("Phase 1 + Phase 2 merge complete.\n")
cat("master_dataset.parquet is the FINAL analytical file.\n")
cat("Next: Phase 3 variable construction and Phase 4 econometrics.\n")
cat(strrep("=", 60), "\n", sep = "")
