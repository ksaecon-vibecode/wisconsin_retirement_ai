# ============================================================
# Phase 4 — Script 04b: IV-Probit Models
# ============================================================
# Project: Wisconsin Retirement AI
# Script:  code/04_econometrics/04b_iv_probit.R
# Author:  Khawaja Sazzad Ali
# Date:    Summer 2026
#
# WHAT THIS SCRIPT DOES:
#   Estimates IV-Probit specifications to address endogeneity
#   of the AI density variable and financial literacy.
#
# TWO ENDOGENEITY CONCERNS:
#
#   1. AI DENSITY ENDOGENEITY:
#      Counties with high AI tool engagement may differ from
#      low-AI counties in ways correlated with enrollment
#      (wealth, education, urbanicity). OLS/Probit estimates
#      would be upward biased.
#      INSTRUMENT: FCC broadband penetration (bb_providers_25_3)
#      Broadband infrastructure is driven by regulatory decisions
#      and construction costs — not by retirement savings behavior.
#      First stage: broadband -> AI density
#
#   2. LITERACY ENDOGENEITY:
#      Workers who seek financial education may also be
#      predisposed to enroll (selection bias).
#      INSTRUMENT: fin_ed_participated (received formal
#      financial education from any source)
#      Exclusion restriction: financial education affects
#      literacy but has no direct effect on enrollment.
#
# APPROACH:
#   Two-stage Probit (Rivers & Vuong 1988 control function approach)
#   implemented via ivreg (2SLS) as linear probability approximation
#   for interpretability, then IV-Probit via AER package.
#
# INPUT:  data/final/master_analytical.parquet
# OUTPUT: outputs/tables/table3_iv_probit.csv
#         docs/phase4_iv_log.txt
# ============================================================

library(tidyverse)
library(arrow)
library(here)
library(glue)
library(fixest)
library(AER)           # ivreg for IV estimation
library(modelsummary)

# ── 0. Setup ─────────────────────────────────────────────────
final_dir <- here("data", "final")
tbl_dir   <- here("outputs", "tables")
docs_dir  <- here("docs")
dir.create(tbl_dir, showWarnings = FALSE, recursive = TRUE)

log_lines <- c()
log <- function(msg) { cat(msg, "\n"); log_lines <<- c(log_lines, msg) }

log(strrep("=", 60))
log("Phase 4 — IV-Probit Models")
log(format(Sys.time()))
log(strrep("=", 60))

# ── 1. Load data ──────────────────────────────────────────────
log("\n[1] Loading analytical dataset...")
df <- read_parquet(file.path(final_dir, "master_analytical.parquet")) %>%
  mutate(
    wave_fe     = factor(SURVEY_WAVE),
    county_fips = as.character(county_fips),
    enrolled    = as.integer(enrolled)
  )

# AI subsample for IV models
df_ai <- df %>%
  filter(
    !is.na(enrolled),
    !is.na(literacy_score),
    !is.na(impatience_index),
    !is.na(ai_density_log),
    !is.na(distress_index),
    !is.na(bb_providers_25_3),
    !is.na(log_median_income),
    !is.na(poverty_rate),
    !is.na(pct_bach_plus),
    !is.na(unemp_rate)
  )
log(glue("  AI subsample: {format(nrow(df_ai), big.mark=',')} obs"))

# IV sample: also needs financial education instrument
df_iv <- df_ai %>%
  filter(!is.na(fin_ed_participated))
log(glue("  IV subsample (with fin_ed): {format(nrow(df_iv), big.mark=',')} obs"))


# ── 2. First stage: Broadband -> AI density ──────────────────
log("\n[2] First stage — Broadband IV for AI density...")

first_stage <- feols(
  ai_density_log ~
    bb_providers_25_3 +
    log_median_income +
    poverty_rate +
    pct_bach_plus +
    unemp_rate |
    wave_fe,
  data    = df_ai,
  cluster = ~county_fips
)

log("  First stage results (broadband -> AI density):")
fs_coefs <- coef(first_stage)
fs_ses   <- sqrt(diag(vcov(first_stage)))
bb_coef  <- fs_coefs["bb_providers_25_3"]
bb_se    <- fs_ses["bb_providers_25_3"]
bb_t     <- bb_coef / bb_se
log(glue("    bb_providers_25_3: b={round(bb_coef,4)}, SE={round(bb_se,4)}, t={round(bb_t,2)}"))

# F-statistic for weak instrument test
# Rule of thumb: F > 10 indicates strong instrument
fs_r2   <- fitstat(first_stage, "r2")$r2
fs_wald <- fitstat(first_stage, "wald")
log(glue("    First stage R²: {round(fs_r2, 3)}"))
log(glue("    Instrument direction: {if_else(bb_coef > 0, 'Positive (expected)', 'Negative (unexpected)')}"))

if (abs(bb_t) > 3.16) {
  log("    Instrument strength: STRONG (|t| > 3.16, equivalent F > 10)")
} else if (abs(bb_t) > 2.00) {
  log("    Instrument strength: MODERATE (|t| > 2)")
} else {
  log("    Instrument strength: WEAK — interpret IV results with caution")
}

# Add fitted AI density to dataset (for control function approach)
df_ai$ai_density_hat   <- fitted(first_stage)
df_ai$ai_density_resid <- residuals(first_stage)


# ── 3. Control function approach (Rivers & Vuong) ─────────────
log("\n[3] IV-Probit via control function approach...")
log("  (Rivers & Vuong 1988: include first-stage residuals in Probit)")

# Model 2 IV: add first-stage residual to address endogeneity
# If residual coefficient is significant -> endogeneity confirmed
m2_iv_cf <- feglm(
  enrolled ~
    literacy_score +
    impatience_index +
    ai_density_log +
    lit_x_ai +
    ai_density_resid +      # Control function term
    distress_index +
    log_median_income +
    poverty_rate +
    pct_bach_plus +
    unemp_rate |
    wave_fe,
  data    = df_ai,
  family  = binomial("probit"),
  cluster = ~county_fips
)

resid_coef <- coef(m2_iv_cf)["ai_density_resid"]
resid_se   <- sqrt(diag(vcov(m2_iv_cf)))["ai_density_resid"]
resid_t    <- resid_coef / resid_se

log(glue("  Control function term (ai_density_resid): b={round(resid_coef,4)}, t={round(resid_t,2)}"))
if (abs(resid_t) > 1.96) {
  log("  -> Significant: endogeneity of AI density CONFIRMED")
  log("     IV estimates correct for upward bias in OLS Probit")
} else {
  log("  -> Not significant: endogeneity not detected at 5% level")
  log("     OLS Probit estimates may be reliable")
}

# Model 3 IV: add impatience x AI interaction
m3_iv_cf <- feglm(
  enrolled ~
    literacy_score +
    impatience_index +
    ai_density_log +
    lit_x_ai +
    imp_x_ai +
    ai_density_resid +
    distress_index +
    log_median_income +
    poverty_rate +
    pct_bach_plus +
    unemp_rate |
    wave_fe,
  data    = df_ai,
  family  = binomial("probit"),
  cluster = ~county_fips
)

log(glue("  Model 3 IV: N={m3_iv_cf$nobs}"))


# ── 4. 2SLS Linear Probability Model (for comparison) ─────────
log("\n[4] 2SLS Linear Probability Model (IV, for robustness)...")

# 2SLS is easier to implement with standard tools and gives
# consistent estimates under weaker distributional assumptions.
# We report this alongside IV-Probit as a robustness check.

m2_2sls <- tryCatch({
  ivreg(
    enrolled ~
      literacy_score +
      impatience_index +
      ai_density_log +
      lit_x_ai +
      distress_index +
      log_median_income +
      poverty_rate +
      pct_bach_plus +
      unemp_rate +
      factor(SURVEY_WAVE) |
      # Instruments: replace ai_density_log and lit_x_ai with
      # bb_providers_25_3 and bb_providers_25_3 * literacy_score
      literacy_score +
      impatience_index +
      bb_providers_25_3 +
      I(bb_providers_25_3 * literacy_score) +
      distress_index +
      log_median_income +
      poverty_rate +
      pct_bach_plus +
      unemp_rate +
      factor(SURVEY_WAVE),
    data = df_ai
  )
}, error = function(e) {
  log(glue("  2SLS failed: {e$message}"))
  NULL
})

if (!is.null(m2_2sls)) {
  log(glue("  2SLS Model 2: N={nrow(df_ai)}"))
  # Diagnostic statistics
  tryCatch({
    diag <- summary(m2_2sls, diagnostics = TRUE)
    log("  2SLS diagnostics:")
    log(glue("    Weak instruments: F={round(diag$diagnostics['Weak instruments','statistic'],2)}, ",
             "p={round(diag$diagnostics['Weak instruments','p-value'],4)}"))
    log(glue("    Wu-Hausman: F={round(diag$diagnostics['Wu-Hausman','statistic'],2)}, ",
             "p={round(diag$diagnostics['Wu-Hausman','p-value'],4)}"))
  }, error = function(e) {
    log("  (Diagnostics not available)")
  })
}


# ── 5. Literacy endogeneity — Financial education IV ──────────
log("\n[5] Financial literacy endogeneity test...")
log("  Instrument: fin_ed_participated (received formal financial education)")

# First stage: fin_ed -> literacy
lit_first_stage <- feols(
  literacy_score ~
    fin_ed_participated +
    log_median_income +
    poverty_rate +
    pct_bach_plus +
    unemp_rate |
    wave_fe,
  data    = df_iv,
  cluster = ~county_fips
)

fed_coef <- coef(lit_first_stage)["fin_ed_participated"]
fed_se   <- sqrt(diag(vcov(lit_first_stage)))["fin_ed_participated"]
fed_t    <- fed_coef / fed_se
log(glue("  fin_ed_participated -> literacy: b={round(fed_coef,4)}, t={round(fed_t,2)}"))

if (fed_t > 1.96) {
  log("  -> Positive and significant: financial education raises literacy [OK]")
} else {
  log("  -> Weak first stage: financial education IV may be invalid")
}

# Probit with literacy instrumented by financial education
df_iv$lit_resid <- residuals(lit_first_stage)

m1_lit_iv <- feglm(
  enrolled ~
    literacy_score +
    impatience_index +
    lit_resid +               # Control function for literacy
    log_median_income +
    poverty_rate +
    pct_bach_plus +
    unemp_rate |
    wave_fe,
  data    = df_iv,
  family  = binomial("probit"),
  cluster = ~county_fips
)

lit_resid_coef <- coef(m1_lit_iv)["lit_resid"]
lit_resid_t    <- lit_resid_coef /
  sqrt(diag(vcov(m1_lit_iv)))["lit_resid"]

log(glue("  Literacy endogeneity test (control function residual):"))
log(glue("    lit_resid: b={round(lit_resid_coef,4)}, t={round(lit_resid_t,2)}"))
if (abs(lit_resid_t) > 1.96) {
  log("  -> Significant: literacy endogeneity CONFIRMED — use IV estimates")
} else {
  log("  -> Not significant: OLS literacy coefficient likely unbiased")
}


# ── 6. IV results summary table ───────────────────────────────
log("\n[6] Producing IV results table...")

iv_models <- list(
  "OLS Probit\n(M2 Baseline)"  = NULL,  # Loaded from 04a
  "CF Probit\n(IV for AI)"     = m2_iv_cf,
  "CF Probit\n(IV M3)"         = m3_iv_cf,
  "LPM IV\n(2SLS)"             = m2_2sls
)

# Remove NULLs
iv_models <- iv_models[!sapply(iv_models, is.null)]

coef_map_iv <- c(
  "literacy_score"    = "Financial literacy score",
  "impatience_index"  = "Impatience index",
  "ai_density_log"    = "Log AI complaint density",
  "lit_x_ai"          = "Literacy × AI density",
  "imp_x_ai"          = "Impatience × AI density",
  "ai_density_resid"  = "First-stage residual (AI)",
  "lit_resid"         = "First-stage residual (literacy)"
)

tryCatch({
  tbl_iv <- modelsummary(
    iv_models,
    coef_map = coef_map_iv,
    stars    = c("*"=0.1, "**"=0.05, "***"=0.01),
    title    = "Table 3: IV-Probit Robustness — Addressing AI Density Endogeneity",
    notes    = paste(
      "Control function (CF) approach instruments AI density with",
      "FCC broadband penetration (25/3 Mbps ISP count).",
      "Significance of first-stage residual indicates endogeneity.",
      "* p<0.10, ** p<0.05, *** p<0.01"
    ),
    output   = "dataframe"
  )

  csv_path <- file.path(tbl_dir, "table3_iv_probit.csv")
  write_csv(tbl_iv, csv_path)
  log(glue("  IV table saved: {csv_path}"))

}, error = function(e) {
  log(glue("  modelsummary failed: {e$message}"))
})


# ── 7. Save log ───────────────────────────────────────────────
log_path <- file.path(docs_dir, "phase4_iv_log.txt")
writeLines(log_lines, log_path)

log(strrep("=", 60))
log("Script 04b complete.")
log("Next: Run 04c_robustness.R for robustness checks")
log(strrep("=", 60))
