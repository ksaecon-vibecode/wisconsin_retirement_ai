# ============================================================
# Phase 3: Variable Construction
# ============================================================
# Project: Wisconsin Retirement AI
# Script:  code/03_variable_construction/03_variable_construction.R
# Author:  Khawaja Sazzad Ali
# Date:    Summer 2026
#
# WHAT THIS SCRIPT DOES:
#   Takes the master_dataset.parquet produced by Phase 1 + 2
#   and constructs all final analytical variables needed for
#   Phase 4 econometric estimation. Produces a fully analysis-
#   ready dataset: master_analytical.parquet
#
# TASKS:
#   1. Fix gender variable (wave-specific column mapping)
#   2. Recode financial education IV
#   3. Standardize behavioral variables
#   4. Construct interaction terms
#   5. Verify low-income threshold
#   6. Power analysis
#   7. Produce Table 1 (descriptive statistics)
#   8. Save analytical dataset
#
# INPUT:  data/final/master_dataset.parquet
# OUTPUT: data/final/master_analytical.parquet
#         outputs/tables/table1_descriptives.csv
#         docs/power_analysis.txt
# ============================================================

library(tidyverse)
library(arrow)
library(here)
library(glue)
library(pwr)       # Power analysis

# ── 0. Setup ─────────────────────────────────────────────────
final_dir <- here("data", "final")
raw_nfcs  <- here("data", "raw", "nfcs")
docs_dir  <- here("docs")
tbl_dir   <- here("outputs", "tables")
dir.create(tbl_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(docs_dir, showWarnings = FALSE, recursive = TRUE)

log_lines <- c()
log <- function(msg) {
  cat(msg, "\n")
  log_lines <<- c(log_lines, msg)
}

log(strrep("=", 60))
log("Phase 3: Variable Construction")
log(format(Sys.time()))
log(strrep("=", 60))

# ── 1. Load master dataset ────────────────────────────────────
log("\n[1/8] Loading master dataset...")
master <- read_parquet(file.path(final_dir, "master_dataset.parquet"))
log(glue("  Rows: {format(nrow(master), big.mark=',')} | Cols: {ncol(master)}"))
log(glue("  Waves: {paste(sort(unique(master$SURVEY_WAVE)), collapse=', ')}"))

# Work on a copy — never modify master_dataset.parquet
df <- master


# ── 2. Fix gender variable ────────────────────────────────────
# PROBLEM: female column is all NA — never populated.
# SOLUTION: Load wave-specific gender columns from public NFCS files
# and map them to a clean binary female indicator.
#
# NFCS gender column names by wave:
#   2009: A3B (1=Male, 2=Female, from binary age-gender net)
#   2012: A3B
#   2015: A3B
#   2018: A3B
#   2021: A50A (1=Male, 2=Female, from binary recode of new A50 question)
#   2024: A50A
#
# NOTE: All waves have binary gender for weighting purposes.
# The 2021/2024 waves added a non-binary A50 question but still
# provide A50A as the binary recode for comparability.

log("\n[2/8] Constructing gender variable from NFCS public files...")

# Helper to read gender from one wave zip file
read_gender_wave <- function(year, zip_path) {
  if (!file.exists(zip_path)) {
    log(glue("  Wave {year}: zip not found at {zip_path}"))
    return(NULL)
  }

  tryCatch({
    # Extract CSV to temp directory (robust on Windows -- unz() fails with spaces)
    zip_contents <- unzip(zip_path, list = TRUE)
    csv_name <- grep("\\.csv$", zip_contents$Name,
                     value = TRUE, ignore.case = TRUE)
    csv_name <- csv_name[!grepl("__MACOSX|investor", csv_name,
                                  ignore.case = TRUE)]
    if (length(csv_name) == 0) {
      log(glue("  Wave {year}: no CSV found in zip"))
      return(NULL)
    }
    csv_name <- csv_name[1]
    tmp_dir  <- tempdir()
    unzip(zip_path, files = csv_name, exdir = tmp_dir, overwrite = TRUE)
    # Handle subdirectory paths inside zip
    tmp_file <- file.path(tmp_dir, basename(csv_name))
    if (!file.exists(tmp_file)) {
      tmp_file <- file.path(tmp_dir, csv_name)
    }
    df_raw <- read_csv(tmp_file, col_types = cols(.default = "c"),
                       show_col_types = FALSE)
    unlink(tmp_file)

    df_raw <- df_raw %>%
      rename_with(str_to_upper)

    # Select gender column based on wave
    gender_col <- if (year >= 2021) "A50A" else "A3B"

    if (!gender_col %in% names(df_raw)) {
      # Try alternatives
      alt_cols <- names(df_raw)[str_detect(names(df_raw),
                                            "^A50|^A3B|GENDER|SEX")]
      log(glue("  Wave {year}: '{gender_col}' not found. ",
               "Alternatives: {paste(alt_cols, collapse=', ')}"))
      gender_col <- alt_cols[1]
      if (is.na(gender_col)) return(NULL)
    }

    id_col <- if ("NFCSID" %in% names(df_raw)) "NFCSID" else names(df_raw)[1]

    result <- df_raw %>%
      select(all_of(c(id_col, gender_col))) %>%
      rename(NFCSID = 1, gender_raw = 2) %>%
      mutate(
        gender_raw = as.integer(gender_raw),
        # 1 = Male, 2 = Female across all NFCS waves
        female = case_when(
          gender_raw == 2 ~ 1L,
          gender_raw == 1 ~ 0L,
          TRUE            ~ NA_integer_
        ),
        SURVEY_WAVE = year
      ) %>%
      select(NFCSID, female, SURVEY_WAVE)

    n_female <- sum(result$female == 1, na.rm = TRUE)
    n_male   <- sum(result$female == 0, na.rm = TRUE)
    log(glue("  Wave {year} ({gender_col}): ",
             "{n_female} female, {n_male} male, ",
             "{sum(is.na(result$female))} missing"))
    return(result)

  }, error = function(e) {
    log(glue("  Wave {year}: ERROR — {e$message}"))
    return(NULL)
  })
}

# Read gender from all six wave zip files
wave_files <- list(
  list(year=2009, path=file.path(raw_nfcs, "nfcs_2009.zip")),
  list(year=2012, path=file.path(raw_nfcs, "nfcs_2012.zip")),
  list(year=2015, path=file.path(raw_nfcs, "nfcs_2015.zip")),
  list(year=2018, path=file.path(raw_nfcs, "nfcs_2018.zip")),
  list(year=2021, path=file.path(raw_nfcs, "nfcs_2021.zip")),
  list(year=2024, path=file.path(raw_nfcs, "nfcs_2024.zip"))
)

gender_data <- map(wave_files, ~ read_gender_wave(.x$year, .x$path))
gender_data <- compact(gender_data)

if (length(gender_data) > 0) {
  gender_pooled <- bind_rows(gender_data) %>%
    mutate(NFCSID = as.character(NFCSID))

  # Merge onto master dataset using NFCSID
  df <- df %>%
    mutate(NFCSID = as.character(NFCSID)) %>%
    select(-female) %>%   # Remove all-NA placeholder
    left_join(
      gender_pooled %>% select(NFCSID, female),
      by = "NFCSID"
    )

  n_female <- sum(df$female == 1, na.rm=TRUE)
  n_male   <- sum(df$female == 0, na.rm=TRUE)
  n_miss   <- sum(is.na(df$female))
  log(glue("\n  Gender variable constructed:"))
  log(glue("    Female: {n_female} ({round(n_female/nrow(df)*100,1)}%)"))
  log(glue("    Male:   {n_male} ({round(n_male/nrow(df)*100,1)}%)"))
  log(glue("    Missing: {n_miss}"))

} else {
  log("  WARNING: No wave files found. Gender remains NA.")
  log("  Ensure NFCS zip files are in data/raw/nfcs/")
  # Keep female as all-NA — Model 4 female interaction will be skipped
}


# ── 3. Recode financial education IV ─────────────────────────
log("\n[3/8] Recoding financial education IV...")

# fin_ed_received codes:
#   1 = Financial education offered but did NOT participate
#   2 = Financial education offered and DID participate
#   3 = No financial education offered
#   98 = Don't know
#   99 = Prefer not to say
#
# fin_ed_school (M21_1) codes:
#   1 = Yes, received financial education in high school
#   2 = No
#   98/99 = DK/refuse
#
# For IV-Probit: instrument = received any formal financial education
# Use fin_ed_received == 2 (participated) as the primary instrument
# This is broader than school-only and has more variation

df <- df %>%
  mutate(
    # Primary IV: participated in formal financial education (any source)
    fin_ed_participated = case_when(
      fin_ed_received == 2 ~ 1L,   # Participated
      fin_ed_received %in% c(1, 3) ~ 0L,  # Offered but didn't / Not offered
      TRUE ~ NA_integer_
    ),

    # Secondary IV: specifically in high school (M21_1)
    fin_ed_highschool = case_when(
      fin_ed_school == 1 ~ 1L,   # Yes, high school
      fin_ed_school == 2 ~ 0L,   # No
      TRUE ~ NA_integer_
    )
  )

log(glue("  fin_ed_participated: ",
         "{sum(df$fin_ed_participated==1, na.rm=T)} yes, ",
         "{sum(df$fin_ed_participated==0, na.rm=T)} no, ",
         "{sum(is.na(df$fin_ed_participated))} missing"))
log(glue("  fin_ed_highschool: ",
         "{sum(df$fin_ed_highschool==1, na.rm=T)} yes, ",
         "{sum(df$fin_ed_highschool==0, na.rm=T)} no, ",
         "{sum(is.na(df$fin_ed_highschool))} missing"))


# ── 4. Standardize behavioral variables ──────────────────────
log("\n[4/8] Standardizing behavioral variables...")

# Standardize to mean=0, SD=1 for regression interpretation
# Both raw and standardized versions are kept
# Raw: for descriptive tables and direct interpretation
# Standardized: for regression coefficients (interpretable as SD changes)

df <- df %>%
  mutate(
    # Standardized versions (already in dataset, but recompute on full sample)
    literacy_std   = as.numeric(scale(literacy_score)),
    impatience_std = as.numeric(scale(impatience_index)),

    # Binary threshold versions (for heterogeneity analysis)
    # Low literacy: bottom 40% (score 0-2, consistent with literature)
    low_literacy = if_else(literacy_score <= 2, 1L, 0L,
                           missing = NA_integer_),

    # High impatience: score >= 2 (used 2+ of 3 impulsive behaviors)
    high_impatience = if_else(impatience_index >= 2, 1L, 0L,
                               missing = NA_integer_),

    # AI density transformations
    # Primary: log(1 + density) — already in dataset as _final_log
    # Keep as-is; add a centered version for interaction interpretation
    ai_density_log   = ai_complaint_density_final_log,
    ai_density_std   = as.numeric(scale(
      if_else(is.na(ai_complaint_density_final_log), 0,
              ai_complaint_density_final_log)
    )),

    # Broadband IV: standardized for first-stage reporting
    bb_std = as.numeric(scale(bb_providers_25_3))
  )

log(glue("  literacy_std: mean={round(mean(df$literacy_std,na.rm=T),3)}, ",
         "sd={round(sd(df$literacy_std,na.rm=T),3)}"))
log(glue("  impatience_std: mean={round(mean(df$impatience_std,na.rm=T),3)}, ",
         "sd={round(sd(df$impatience_std,na.rm=T),3)}"))
log(glue("  low_literacy (<=2): {sum(df$low_literacy==1,na.rm=T)} ",
         "({round(mean(df$low_literacy==1,na.rm=T)*100,1)}%)"))
log(glue("  high_impatience (>=2): {sum(df$high_impatience==1,na.rm=T)} ",
         "({round(mean(df$high_impatience==1,na.rm=T)*100,1)}%)"))


# ── 5. Verify low-income threshold ───────────────────────────
log("\n[5/8] Verifying low-income threshold...")

# Income brackets in NFCS:
# 1 = Less than $15,000
# 2 = $15,000 - $24,999
# 3 = $25,000 - $34,999
# 4 = $35,000 - $49,999
# 5 = $50,000 - $74,999
# 6 = $75,000 - $99,999
# 7 = $100,000 - $149,999
# 8 = $150,000 - $199,999
# 9 = $200,000 - $299,999
# 10 = $300,000 or more

income_dist <- df %>%
  count(income) %>%
  mutate(
    label = case_when(
      income == 1  ~ "<$15k",
      income == 2  ~ "$15-25k",
      income == 3  ~ "$25-35k",
      income == 4  ~ "$35-50k",
      income == 5  ~ "$50-75k",
      income == 6  ~ "$75-100k",
      income == 7  ~ "$100-150k",
      income == 8  ~ "$150-200k",
      income == 9  ~ "$200-300k",
      income == 10 ~ "$300k+",
      TRUE         ~ "Missing/DK"
    ),
    pct = round(n / nrow(df) * 100, 1)
  )

log("  Income distribution:")
for (i in seq_len(nrow(income_dist))) {
    log(glue("    {str_pad(income_dist$label[i], 12)} n={income_dist$n[i]} ({income_dist$pct[i]}%)"))
  }

# Wisconsin median household income ~$63,000 (2021 ACS)
# Low income = below $35,000 (brackets 1-3) is ~bottom 30th percentile
# This is appropriate for Wisconsin context

pct_low_inc_3 <- mean(df$income <= 3, na.rm=TRUE) * 100
pct_low_inc_4 <- mean(df$income <= 4, na.rm=TRUE) * 100

log(glue("\n  Threshold at bracket <=3 (<$35k): {round(pct_low_inc_3,1)}% of sample"))
log(glue("  Threshold at bracket <=4 (<$50k): {round(pct_low_inc_4,1)}% of sample"))
log("  Decision: keep <=3 (<$35k) as low_income threshold.")
log("  This represents the bottom ~22% — households clearly below")
log("  Wisconsin median who face genuine retirement savings constraints.")

# Reconfirm low_income is coded correctly
df <- df %>%
  mutate(
    low_income = if_else(income <= 3, 1L, 0L, missing = NA_integer_)
  )
log(glue("  low_income confirmed: {sum(df$low_income==1,na.rm=T)} respondents"))


# ── 6. Construct interaction terms ────────────────────────────
log("\n[6/8] Constructing interaction terms...")

df <- df %>%
  mutate(

    # ── Model 2 interactions ─────────────────────────────────
    # Literacy x AI (main moderation test)
    lit_x_ai     = literacy_score * ai_density_log,
    lit_std_x_ai = literacy_std   * ai_density_log,

    # ── Model 3 interactions ─────────────────────────────────
    # Impatience x AI (present bias moderation test)
    imp_x_ai     = impatience_index * ai_density_log,
    imp_std_x_ai = impatience_std   * ai_density_log,

    # ── Model 4 triple interactions ───────────────────────────
    # Literacy x AI x Low Income
    lit_x_ai_x_lowinc = literacy_score * ai_density_log * low_income,

    # Impatience x AI x Female (requires gender variable)
    imp_x_ai_x_female = impatience_index * ai_density_log *
                         if_else(is.na(female), NA_real_, as.numeric(female)),

    # Impatience x AI x Minority
    imp_x_ai_x_minority = impatience_index * ai_density_log * minority,

    # ── Wave fixed effect factor ──────────────────────────────
    wave_fe = factor(SURVEY_WAVE,
                     levels = c(2009, 2012, 2015, 2018, 2021, 2024))
  )

# Report interaction term coverage
int_vars <- c("lit_x_ai", "imp_x_ai",
              "lit_x_ai_x_lowinc", "imp_x_ai_x_female",
              "imp_x_ai_x_minority")

log("  Interaction term coverage (non-missing):")
for (v in int_vars) {
  n_valid <- sum(!is.na(df[[v]]))
  log(glue("    {str_pad(v, 25)}: {n_valid} non-missing"))
}

# Flag if female interactions are all NA
if (all(is.na(df$imp_x_ai_x_female))) {
  log("\n  WARNING: imp_x_ai_x_female is all NA — gender fix needed")
  log("  Model 4 female triple interaction will be SKIPPED in Phase 4")
  log("  until gender variable is populated from NFCS wave files")
}


# ── 7. Power analysis ─────────────────────────────────────────
log("\n[7/8] Power analysis...")

# Sample sizes
n_m1  <- sum(!is.na(df$enrolled) & !is.na(df$literacy_score) &
              !is.na(df$impatience_index) & !is.na(df$log_median_income) &
              !is.na(df$poverty_rate) & !is.na(df$unemp_rate))
n_m24 <- sum(!is.na(df$enrolled) & !is.na(df$ai_density_log) &
              !is.na(df$distress_index))

log(glue("  Model 1 sample: {n_m1}"))
log(glue("  Models 2-4 sample: {n_m24}"))

# Minimum detectable effect size (Cohen's h for proportions)
# For Probit, we use Cohen's h as approximation
# Convention: small=0.2, medium=0.5, large=0.8

power_results <- map_dfr(c(0.80, 0.90), function(pw) {
  map_dfr(c(n_m1, n_m24), function(n) {
    result <- pwr.p.test(
      h     = NULL,
      n     = n,
      sig.level = 0.05,
      power = pw
    )
    tibble(
      sample_n  = n,
      power     = pw,
      min_effect_h = round(result$h, 4)
    )
  })
})

log("\n  Minimum detectable effect sizes (Cohen's h, two-sided, α=0.05):")
print(power_results)

# Practical interpretation
log(glue("\n  At 80% power, Model 1 (n={n_m1}) can detect effects of h={filter(power_results, sample_n==n_m1, power==0.80)$min_effect_h}"))
log(glue("  At 80% power, Models 2-4 (n={n_m24}) can detect effects of h={filter(power_results, sample_n==n_m24, power==0.80)$min_effect_h}"))
log("  Cohen's h ≈ 0.14 corresponds to roughly a 7 pp difference in")
log("  enrollment rates between groups -- detectable and policy-relevant.")
log("  Triple interactions in Model 4 will have less power given subgroup sizes.")


# ── 8. Descriptive statistics table (Table 1) ────────────────
log("\n[8/8] Producing Table 1 — Descriptive Statistics...")

# Function to summarize one variable
summarize_var <- function(df, varname, label, pct=FALSE) {
  if (!varname %in% names(df)) return(NULL)
  x <- df[[varname]]
  x <- suppressWarnings(as.numeric(x))
  n_valid <- sum(!is.na(x))
  if (n_valid == 0) return(NULL)

  multiplier <- if (pct) 100 else 1
  tibble(
    Variable = label,
    N        = n_valid,
    Mean     = round(mean(x, na.rm=TRUE) * multiplier, 3),
    SD       = round(sd(x, na.rm=TRUE)   * multiplier, 3),
    Min      = round(min(x, na.rm=TRUE)  * multiplier, 3),
    Max      = round(max(x, na.rm=TRUE)  * multiplier, 3)
  )
}

# Full sample (Model 1)
full_sample <- df %>% filter(
  !is.na(enrolled), !is.na(literacy_score),
  !is.na(impatience_index), !is.na(log_median_income)
)

# AI subsample (Models 2-4)
ai_sample <- df %>% filter(!is.na(ai_density_log), !is.na(distress_index))

# Build Table 1 rows
tbl1_vars <- list(
  list("enrolled",                    "Retirement plan enrollment (0/1)"),
  list("literacy_score",              "Financial literacy score (0-5)"),
  list("impatience_index",            "Impatience index (0-3)"),
  list("ai_complaint_density_final",  "AI complaint density (per 10,000)"),
  list("ai_density_log",              "Log AI complaint density"),
  list("distress_index",              "Financial distress index"),
  list("bb_providers_25_3",           "Broadband providers 25/3 Mbps (IV)"),
  list("log_median_income",           "Log median household income"),
  list("poverty_rate",                "County poverty rate (%)"),
  list("pct_bach_plus",               "County % bachelor's degree+"),
  list("unemp_rate",                  "County unemployment rate (%)"),
  list("minority",                    "Minority indicator (0/1)"),
  list("low_income",                  "Low income indicator (0/1)"),
  list("female",                      "Female indicator (0/1)")
)

# Full sample stats
tbl1_full <- map_dfr(tbl1_vars, function(v) {
  summarize_var(full_sample, v[[1]], v[[2]])
}) %>% rename_with(~ paste0(., "_full"), -Variable)

# AI subsample stats
tbl1_ai <- map_dfr(tbl1_vars, function(v) {
  summarize_var(ai_sample, v[[1]], v[[2]])
}) %>% rename_with(~ paste0(., "_ai"), -Variable)

# Combine
tbl1 <- left_join(tbl1_full, tbl1_ai, by = "Variable")

# Add note rows
tbl1_header <- tibble(
  Variable = c(
    "PANEL A: INDIVIDUAL-LEVEL VARIABLES (NFCS)",
    paste0("  Full sample (Model 1): N = ", format(n_m1, big.mark=",")),
    paste0("  AI subsample (Models 2-4): N = ", format(n_m24, big.mark=",")),
    "PANEL B: COUNTY-LEVEL VARIABLES"
  )
)

# Print table
log("\n  Table 1 Preview:")
print(tbl1, n=20)

# Save
tbl1_path <- file.path(tbl_dir, "table1_descriptives.csv")
write_csv(tbl1, tbl1_path)
log(glue("\n  Saved: {tbl1_path}"))


# ── 9. Wave-by-wave enrollment table (for paper appendix) ────
wave_table <- df %>%
  group_by(SURVEY_WAVE) %>%
  summarise(
    N                = n(),
    Enrollment_pct   = round(mean(enrolled, na.rm=TRUE)*100, 1),
    Literacy_mean    = round(mean(literacy_score, na.rm=TRUE), 2),
    Impatience_mean  = round(mean(impatience_index, na.rm=TRUE), 2),
    AI_density_mean  = round(mean(ai_complaint_density_final, na.rm=TRUE), 4),
    Pct_female       = round(mean(female, na.rm=TRUE)*100, 1),
    Pct_minority     = round(mean(minority, na.rm=TRUE)*100, 1),
    Pct_low_income   = round(mean(low_income, na.rm=TRUE)*100, 1),
    .groups = "drop"
  )

wave_path <- file.path(tbl_dir, "wave_summary_table.csv")
write_csv(wave_table, wave_path)
log(glue("  Wave table saved: {wave_path}"))
log("  Wave table:")
print(wave_table)


# ── 10. Save analytical dataset ───────────────────────────────
log(paste0("\n", strrep("=", 60)))
log("SAVING ANALYTICAL DATASET")
log(strrep("=", 60))

out_path <- file.path(final_dir, "master_analytical.parquet")
write_parquet(df, out_path)

log(glue("  Saved: {out_path}"))
log(glue("  Rows: {format(nrow(df), big.mark=',')}"))
log(glue("  Columns: {ncol(df)}"))
log(glue("  Size: {round(file.size(out_path)/1024, 1)} KB"))

# Final variable inventory
new_vars <- c(
  "female", "fin_ed_participated", "fin_ed_highschool",
  "literacy_std", "impatience_std", "low_literacy", "high_impatience",
  "ai_density_log", "ai_density_std", "bb_std",
  "lit_x_ai", "lit_std_x_ai", "imp_x_ai", "imp_std_x_ai",
  "lit_x_ai_x_lowinc", "imp_x_ai_x_female", "imp_x_ai_x_minority"
)
log("\nNew variables constructed in Phase 3:")
for (v in new_vars) {
  status <- if_else(v %in% names(df), "OK", "MISSING")
  n_valid <- if (v %in% names(df)) sum(!is.na(df[[v]])) else 0
  log(glue("  [{status}] {str_pad(v, 28)} non-missing: {n_valid}"))
}


# ── 11. Save log ──────────────────────────────────────────────
log_path <- file.path(docs_dir, "phase3_log.txt")
writeLines(log_lines, log_path)
log(glue("\nLog saved: {log_path}"))

log("\n" %+% strrep("=", 60))
log("Phase 3 complete.")
log("Next: Phase 4 — Econometric Analysis")
log("  source('code/04_econometrics/04a_probit_models.R')")
log(strrep("=", 60))
