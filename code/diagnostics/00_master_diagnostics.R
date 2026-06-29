# ============================================================
# Master Dataset Diagnostics
# Run before Phase 3 to confirm data integrity
# ============================================================
# Project: Wisconsin Retirement AI
# Script:  code/diagnostics/00_master_diagnostics.R
# Author:  Khawaja Sazzad Ali
# Date:    Summer 2026
#
# WHAT THIS CHECKS:
#   1. Variable distributions — are ranges plausible?
#   2. Missing data patterns — are missings random or systematic?
#   3. Wave balance — sample sizes and key means by wave
#   4. County coverage — which counties are represented?
#   5. Outcome variable — enrollment rate patterns
#   6. Key behavioral variables — literacy and impatience
#   7. AI density — distribution and variation
#   8. Correlation matrix — multicollinearity red flags
#   9. Regression sample sizes — how many obs per model?
#  10. Gender variable — does it exist and is it coded correctly?
#
# OUTPUT:
#   docs/diagnostics_report.txt
#   outputs/figures/diag_distributions.png
#   outputs/figures/diag_correlations.png
# ============================================================

library(tidyverse)
library(arrow)
library(here)
library(glue)

# ── 0. Setup ─────────────────────────────────────────────────
final_dir <- here("data", "final")
docs_dir  <- here("docs")
fig_dir   <- here("outputs", "figures")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

log_lines <- c()
log <- function(msg) {
  cat(msg, "\n")
  log_lines <<- c(log_lines, msg)
}

log(strrep("=", 60))
log("Master Dataset Diagnostics")
log(format(Sys.time()))
log(strrep("=", 60))

# ── 1. Load master dataset ────────────────────────────────────
log("\n[1] Loading master dataset...")
master <- read_parquet(file.path(final_dir, "master_dataset.parquet"))
log(glue("  Rows: {format(nrow(master), big.mark=',')}"))
log(glue("  Columns: {ncol(master)}"))
log(glue("  Column names: {paste(sort(names(master)), collapse=', ')}"))


# ── 2. Variable distributions ─────────────────────────────────
log("\n[2] Variable distributions (key regression variables):")

check_var <- function(df, varname, expected_min, expected_max, label) {
  if (!varname %in% names(df)) {
    log(glue("  {label}: NOT FOUND IN DATASET"))
    return(invisible(NULL))
  }
  x <- df[[varname]]
  n_valid <- sum(!is.na(x))
  n_miss  <- sum(is.na(x))
  rng_ok  <- min(x, na.rm=TRUE) >= expected_min &
              max(x, na.rm=TRUE) <= expected_max
  flag <- if_else(rng_ok, "OK", "RANGE FLAG")
  log(glue(
    "  {label}:\n",
    "    n={format(n_valid, big.mark=',')}, missing={n_miss} ",
    "({round(n_miss/nrow(df)*100,1)}%)\n",
    "    mean={round(mean(x,na.rm=T),3)}, sd={round(sd(x,na.rm=T),3)}\n",
    "    min={round(min(x,na.rm=T),3)}, max={round(max(x,na.rm=T),3)}\n",
    "    expected range [{expected_min}, {expected_max}]: {flag}"
  ))
}

check_var(master, "enrolled",                  0, 1,    "Enrolled (outcome)")
check_var(master, "literacy_score",            0, 5,    "Literacy score")
check_var(master, "impatience_index",          0, 3,    "Impatience index")
check_var(master, "ai_complaint_density_final",0, 10,   "AI density (primary)")
check_var(master, "distress_index",           -1, 1,    "Distress index")
check_var(master, "bb_providers_25_3",         0, 10,   "Broadband providers (IV)")
check_var(master, "poverty_rate",              0, 100,  "Poverty rate")
check_var(master, "log_median_income",         9, 13,   "Log median income")
check_var(master, "pct_bach_plus",             0, 100,  "Pct bachelor+")
check_var(master, "unemp_rate",                0, 30,   "Unemployment rate")
check_var(master, "minority",                  0, 1,    "Minority indicator")
check_var(master, "low_income",                0, 1,    "Low income indicator")


# ── 3. Missing data patterns ──────────────────────────────────
log("\n[3] Missing data patterns:")

key_vars <- c(
  "enrolled", "literacy_score", "impatience_index",
  "county_fips", "SURVEY_WAVE",
  "bb_providers_25_3", "poverty_rate", "unemp_rate",
  "log_median_income", "pct_bach_plus",
  "minority", "low_income",
  "ai_complaint_density_final", "distress_index"
)

miss_df <- tibble(
  variable  = key_vars,
  n_missing = sapply(key_vars, function(v) {
    if (v %in% names(master)) sum(is.na(master[[v]])) else NA_integer_
  }),
  pct_miss  = round(n_missing / nrow(master) * 100, 1)
)

log("  Variable                          N Missing   Pct Miss")
log("  " %+% strrep("-", 50))
for (i in seq_len(nrow(miss_df))) {
  flag <- if_else(miss_df$pct_miss[i] > 50, " *** HIGH ***", "")
  log(glue("  {str_pad(miss_df$variable[i], 30, 'right')}  ",
           "{str_pad(miss_df$n_missing[i], 8, 'left')}   ",
           "{str_pad(miss_df$pct_miss[i], 6, 'left')}%{flag}"))
}

# Check if missingness on AI density is explainable by wave
log("\n  Missingness on AI density by wave:")
master %>%
  group_by(SURVEY_WAVE) %>%
  summarise(
    n = n(),
    n_ai_missing = sum(is.na(ai_complaint_density_final)),
    pct_missing  = round(mean(is.na(ai_complaint_density_final))*100, 1),
    .groups = "drop"
  ) %>%
  walk(function(row) {}) # suppress output, use print below
master %>%
  group_by(SURVEY_WAVE) %>%
  summarise(
    n            = n(),
    n_ai_missing = sum(is.na(ai_complaint_density_final)),
    pct_missing  = round(mean(is.na(ai_complaint_density_final))*100, 1),
    .groups = "drop"
  ) %>%
  print()


# ── 4. Wave balance ───────────────────────────────────────────
log("\n[4] Wave balance:")
wave_summary <- master %>%
  group_by(SURVEY_WAVE) %>%
  summarise(
    n              = n(),
    enroll_pct     = round(mean(enrolled, na.rm=TRUE)*100, 1),
    mean_literacy  = round(mean(literacy_score, na.rm=TRUE), 2),
    mean_impatience= round(mean(impatience_index, na.rm=TRUE), 2),
    mean_ai        = round(mean(ai_complaint_density_final, na.rm=TRUE), 4),
    pct_minority   = round(mean(minority, na.rm=TRUE)*100, 1),
    pct_low_income = round(mean(low_income, na.rm=TRUE)*100, 1),
    .groups = "drop"
  )
print(wave_summary, n=10)

# Flag: are enrollment trends monotonically declining (expected)?
enroll_trend <- wave_summary %>%
  filter(SURVEY_WAVE < 2024) %>%
  pull(enroll_pct)
if (all(diff(enroll_trend) <= 2)) {
  log("  Enrollment trend: consistent decline 2009-2021 [OK]")
} else {
  log("  Enrollment trend: non-monotonic — investigate")
}


# ── 5. Outcome variable checks ────────────────────────────────
log("\n[5] Outcome variable — retirement enrollment:")
log(glue("  Overall enrollment rate: {round(mean(master$enrolled)*100,1)}%"))
log(glue("  N enrolled: {sum(master$enrolled)}"))
log(glue("  N not enrolled: {sum(master$enrolled == 0)}"))

# Check for extreme county-level variation
if ("county_fips" %in% names(master)) {
  county_enroll <- master %>%
    group_by(county_fips) %>%
    summarise(
      n        = n(),
      enroll   = mean(enrolled, na.rm=TRUE),
      .groups  = "drop"
    ) %>%
    filter(n >= 5)  # Only counties with >= 5 respondents

  log(glue("  Counties with >= 5 respondents: {nrow(county_enroll)}"))
  log(glue("  County enrollment range: {round(min(county_enroll$enroll)*100,1)}% - {round(max(county_enroll$enroll)*100,1)}%"))
  log(glue("  County enrollment SD: {round(sd(county_enroll$enroll)*100,1)} pp"))
}


# ── 6. Behavioral variable checks ─────────────────────────────
log("\n[6] Behavioral variables:")

# Literacy score distribution
lit_dist <- table(master$literacy_score)
log("  Literacy score distribution (0-5):")
log(glue("    {paste(names(lit_dist), collapse='  ')}"))
log(glue("    {paste(as.integer(lit_dist), collapse='  ')}"))

# Floor/ceiling check
pct_zero_lit <- mean(master$literacy_score == 0, na.rm=TRUE)
pct_five_lit <- mean(master$literacy_score == 5, na.rm=TRUE)
log(glue("  Pct scoring 0 (floor): {round(pct_zero_lit*100,1)}%"))
log(glue("  Pct scoring 5 (ceiling): {round(pct_five_lit*100,1)}%"))
if (pct_five_lit > 0.30) {
  log("  WARNING: >30% ceiling — consider alternative specification")
}

# Impatience index distribution
imp_dist <- table(master$impatience_index)
log("\n  Impatience index distribution (0-3):")
log(glue("    {paste(names(imp_dist), collapse='  ')}"))
log(glue("    {paste(as.integer(imp_dist), collapse='  ')}"))


# ── 7. AI density checks ─────────────────────────────────────
log("\n[7] AI density variable:")

ai_sub <- master %>% filter(!is.na(ai_complaint_density_final))
log(glue("  Non-missing: {nrow(ai_sub)} respondents"))
log(glue("  Mean: {round(mean(ai_sub$ai_complaint_density_final),4)}"))
log(glue("  Median: {round(median(ai_sub$ai_complaint_density_final),4)}"))
log(glue("  SD: {round(sd(ai_sub$ai_complaint_density_final),4)}"))
log(glue("  Max: {round(max(ai_sub$ai_complaint_density_final),4)}"))
log(glue("  Pct zero: {round(mean(ai_sub$ai_complaint_density_final == 0)*100,1)}%"))

# Check: is there enough variation for identification?
ai_cv <- sd(ai_sub$ai_complaint_density_final) /
         mean(ai_sub$ai_complaint_density_final)
log(glue("  Coefficient of variation: {round(ai_cv,2)}"))
if (ai_cv > 0.5) {
  log("  Variation: SUFFICIENT for identification [OK]")
} else {
  log("  Variation: LOW — may limit identification power")
}


# ── 8. Correlation matrix (multicollinearity check) ───────────
log("\n[8] Correlation matrix (key regressors):")

corr_vars <- c(
  "enrolled", "literacy_score", "impatience_index",
  "ai_complaint_density_final", "distress_index",
  "bb_providers_25_3", "log_median_income",
  "poverty_rate", "pct_bach_plus", "unemp_rate",
  "minority", "low_income"
)
corr_vars_present <- corr_vars[corr_vars %in% names(master)]

corr_matrix <- master %>%
  select(all_of(corr_vars_present)) %>%
  cor(use = "pairwise.complete.obs") %>%
  round(2)

# Flag high correlations (|r| > 0.7) that could cause multicollinearity
high_corr <- which(abs(corr_matrix) > 0.7 & abs(corr_matrix) < 1,
                   arr.ind = TRUE)
if (nrow(high_corr) > 0) {
  log("  High correlations (|r| > 0.7) — potential multicollinearity:")
  for (i in seq_len(nrow(high_corr))) {
    r <- high_corr[i, ]
    if (r[1] < r[2]) {  # Only show each pair once
      v1 <- rownames(corr_matrix)[r[1]]
      v2 <- colnames(corr_matrix)[r[2]]
      val <- corr_matrix[r[1], r[2]]
      log(glue("    {v1} x {v2}: r = {val}"))
    }
  }
} else {
  log("  No correlations above 0.7 [OK]")
}

# Print full correlation matrix
log("\n  Full correlation matrix:")
print(corr_matrix)


# ── 9. Regression sample sizes ────────────────────────────────
log("\n[9] Expected regression sample sizes:")

# Model 1: all respondents with complete behavioral variables
m1_sample <- master %>%
  filter(!is.na(enrolled), !is.na(literacy_score),
         !is.na(impatience_index), !is.na(log_median_income),
         !is.na(poverty_rate), !is.na(unemp_rate))
log(glue("  Model 1 (behavioral baseline): {format(nrow(m1_sample), big.mark=',')} obs"))

# Models 2-4: also need AI density
m24_sample <- m1_sample %>%
  filter(!is.na(ai_complaint_density_final), !is.na(distress_index))
log(glue("  Models 2-4 (AI moderation): {format(nrow(m24_sample), big.mark=',')} obs"))

# IV sample: also needs school financial education instrument
iv_col <- names(master)[str_detect(names(master), "fin_ed_school|M21_1|m21_1")]
if (length(iv_col) > 0) {
  iv_sample <- m1_sample %>% filter(!is.na(.data[[iv_col[1]]]))
  log(glue("  IV-Probit sample ({iv_col[1]}): {format(nrow(iv_sample), big.mark=',')} obs"))
} else {
  log("  IV instrument (fin_ed_school): NOT FOUND IN DATASET")
  log("  Available columns with 'fin': " %+%
      paste(names(master)[str_detect(names(master), "fin")], collapse=", "))
}

# Subgroup checks
log("\n  Subgroup sizes (for Model 4 triple interactions):")
log(glue("  Low income (low_income=1): {sum(master$low_income==1, na.rm=TRUE)}"))
log(glue("  Minority (minority=1): {sum(master$minority==1, na.rm=TRUE)}"))

# Gender check
gender_col <- names(master)[str_detect(names(master),
              regex("female|gender|sex|A50|a50", ignore_case=TRUE))]
if (length(gender_col) > 0) {
  log(glue("  Gender columns found: {paste(gender_col, collapse=', ')}"))
  for (gc in gender_col[1:min(2, length(gender_col))]) {
    tab <- table(master[[gc]], useNA="always")
    log(glue("  {gc} distribution: {paste(names(tab), as.integer(tab), sep='=', collapse=', ')}"))
  }
} else {
  log("  Gender variable: NOT FOUND — needs to be constructed in Phase 3")
  log("  This is expected — gender requires wave-specific column handling")
}


# ── 10. IV instrument check ───────────────────────────────────
log("\n[10] Instrumental variable checks:")

# FCC broadband (IV for AI exposure)
if ("bb_providers_25_3" %in% names(master)) {
  log("  FCC broadband (IV for AI exposure):")
  log(glue("    Non-missing: {sum(!is.na(master$bb_providers_25_3))}"))
  log(glue("    Mean: {round(mean(master$bb_providers_25_3, na.rm=TRUE), 2)} ISPs"))
  log(glue("    Range: {min(master$bb_providers_25_3, na.rm=TRUE)} - {max(master$bb_providers_25_3, na.rm=TRUE)}"))

  # First-stage correlation (should be positive and meaningful)
  if (!is.na(cor(master$bb_providers_25_3,
                 master$ai_complaint_density_final,
                 use="pairwise.complete.obs"))) {
    r_iv <- cor(master$bb_providers_25_3,
                master$ai_complaint_density_final,
                use="pairwise.complete.obs")
    log(glue("    Correlation with AI density: r = {round(r_iv, 3)}"))
    if (abs(r_iv) > 0.1) {
      log("    First-stage signal: PRESENT [OK]")
    } else {
      log("    First-stage signal: WEAK — IV may be invalid")
    }
  }
}

# School financial education (IV for literacy)
fin_ed_cols <- names(master)[str_detect(names(master), "fin_ed|M21|m21")]
log(glue("\n  Financial education columns: {paste(fin_ed_cols, collapse=', ')}"))
for (fc in fin_ed_cols) {
  tab <- table(master[[fc]], useNA="ifany")
  log(glue("  {fc}: {paste(names(tab), as.integer(tab), sep='=', collapse=', ')}"))
}


# ── 11. Final verdict ─────────────────────────────────────────
log("\n" %+% strrep("=", 60))
log("DIAGNOSTIC VERDICT")
log(strrep("=", 60))

checks <- list(
  "Sample size adequate (n>1000)"   = nrow(m1_sample) > 1000,
  "No zero missing on core vars"    = all(miss_df$n_missing[1:5] == 0),
  "AI density has variation (CV>0.5)"= ai_cv > 0.5,
  "Models 2-4 sample adequate (n>500)"= nrow(m24_sample) > 500,
  "Broadband IV present"            = "bb_providers_25_3" %in% names(master),
  "Minority subgroup adequate (n>100)"= sum(master$minority==1, na.rm=TRUE) > 100
)

all_pass <- TRUE
for (nm in names(checks)) {
  status <- if_else(checks[[nm]], "[PASS]", "[FAIL]")
  log(glue("  {status} {nm}"))
  if (!checks[[nm]]) all_pass <- FALSE
}

log("")
if (all_pass) {
  log("ALL CHECKS PASSED. Proceed to Phase 3.")
} else {
  log("Some checks failed. Review above before Phase 3.")
}


# ── 12. Save report ───────────────────────────────────────────
report_path <- file.path(docs_dir, "diagnostics_report.txt")
writeLines(log_lines, report_path)
cat(glue("\nReport saved: {report_path}\n"))

cat(strrep("=", 60), "\n")
cat("Diagnostics complete.\n")
cat(strrep("=", 60), "\n")
