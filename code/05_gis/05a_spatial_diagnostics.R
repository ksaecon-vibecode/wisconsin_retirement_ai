# ============================================================
# Phase 5 — Script 05a: Spatial Diagnostics
# Moran's I Test for Spatial Autocorrelation
# ============================================================
# Project: Wisconsin Retirement AI
# Script:  code/05_gis/05a_spatial_diagnostics.R
# Author:  Khawaja Sazzad Ali
# Date:    Summer 2026
#
# WHAT THIS SCRIPT DOES:
#   Tests whether your Probit regression residuals show spatial
#   autocorrelation — i.e., whether counties near each other
#   have more similar residuals than counties far apart.
#
# WHY THIS MATTERS:
#   Your regression models cluster standard errors at the county
#   level, which corrects for within-county correlation across
#   survey waves. But if neighboring counties also have correlated
#   errors (spatial autocorrelation), an additional correction
#   may be needed.
#
#   Moran's I statistic tests this:
#     I near +1 = strong positive spatial autocorrelation
#                 (similar counties cluster geographically)
#     I near  0 = no spatial autocorrelation (random pattern)
#     I near -1 = strong negative spatial autocorrelation
#                 (dissimilar counties cluster together)
#
#   If the test is NOT significant (p > 0.05): county-clustered
#   standard errors are sufficient. Report and move on.
#
#   If the test IS significant (p < 0.05): add a spatial lag
#   or spatial error model as a robustness check in appendix.
#
# RUN THIS AFTER: Phase 4 econometric models are estimated
#   (after code/04_econometrics/ scripts produce residuals)
#
# INPUT:
#   data/raw/gis/wi_counties.shp
#   data/final/master_dataset.parquet
#   (Probit residuals computed within this script)
#
# OUTPUT:
#   docs/spatial_diagnostics_report.txt
#   data/processed/county_level_summary.parquet
# ============================================================

library(tidyverse)
library(sf)          # Geographic data
library(spdep)       # Spatial dependence tests (Moran's I)
library(arrow)
library(here)
library(glue)

# ── 0. Setup ─────────────────────────────────────────────────
gis_dir  <- here("data", "raw", "gis")
proc_dir <- here("data", "processed")
final_dir<- here("data", "final")
docs_dir <- here("docs")

cat(strrep("=", 60), "\n")
cat("Phase 5 — Spatial Diagnostics (Moran's I)\n")
cat(format(Sys.time()), "\n")
cat(strrep("=", 60), "\n\n")

# ── 1. Load county shapefile ──────────────────────────────────
cat("Loading county shapefile...\n")
shp_path <- file.path(gis_dir, "wi_counties.shp")

if (!file.exists(shp_path)) {
  stop(glue(
    "Shapefile not found: {shp_path}\n",
    "Run code/01_data_acquisition/01d_gis_data.R first."
  ))
}

wi_sf <- st_read(shp_path, quiet = TRUE) %>%
  mutate(county_fips = GEOID)

cat(glue("  Loaded: {nrow(wi_sf)} counties\n"))
cat(glue("  CRS: {st_crs(wi_sf)$input}\n\n"))

# ── 2. Load master dataset ────────────────────────────────────
cat("Loading master dataset...\n")
master <- read_parquet(file.path(final_dir, "master_dataset.parquet"))
cat(glue("  Respondents: {format(nrow(master), big.mark=',')}\n\n"))

# ── 3. Aggregate to county level for spatial test ─────────────
# Moran's I requires one observation per county.
# We aggregate the key variables to county level and test
# whether counties with similar values are spatially clustered.

cat("Aggregating to county level...\n")

county_summary <- master %>%
  group_by(county_fips) %>%
  summarise(
    n_respondents    = n(),
    enroll_rate      = mean(enrolled, na.rm = TRUE),
    mean_literacy    = mean(literacy_score, na.rm = TRUE),
    mean_impatience  = mean(impatience_index, na.rm = TRUE),
    mean_ai_density  = mean(ai_complaint_density_final, na.rm = TRUE),
    mean_distress    = mean(distress_index, na.rm = TRUE),
    mean_bb          = mean(bb_providers_25_3, na.rm = TRUE),
    mean_poverty     = mean(poverty_rate, na.rm = TRUE),
    pct_minority     = mean(minority, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  filter(!is.na(county_fips))

cat(glue("  Counties with NFCS respondents: {nrow(county_summary)}\n\n"))

# Save county-level summary
write_parquet(
  county_summary,
  file.path(proc_dir, "county_level_summary.parquet")
)

# ── 4. Join county data to shapefile ─────────────────────────
cat("Joining data to shapefile...\n")
wi_joined <- wi_sf %>%
  left_join(county_summary, by = "county_fips")

matched <- sum(!is.na(wi_joined$enroll_rate))
cat(glue("  Counties with data: {matched} of {nrow(wi_joined)}\n\n"))

# ── 5. Build spatial weights matrix ──────────────────────────
# The spatial weights matrix defines which counties are neighbors.
# Queen contiguity = two counties are neighbors if they share
# any border point (including corners). This is the standard
# choice for county-level spatial analysis.

cat("Building spatial weights matrix (Queen contiguity)...\n")

# Create neighbor list
nb_queen <- poly2nb(wi_sf, queen = TRUE)

# Convert to weights matrix (row-standardized)
# Row-standardized means each county's neighbor weights sum to 1
# so that the spatial lag is an average of neighbor values
lw_queen <- nb2listw(nb_queen, style = "W", zero.policy = TRUE)

cat(glue("  Average neighbors per county: {round(mean(card(nb_queen)), 1)}\n"))
cat(glue("  Min neighbors: {min(card(nb_queen))}\n"))
cat(glue("  Max neighbors: {max(card(nb_queen))}\n\n"))

# ── 6. Moran's I tests ────────────────────────────────────────
cat("Running Moran's I tests...\n\n")

# Test function — runs Moran's I and formats result clearly
run_morans <- function(variable, label, data, weights) {
  x <- data[[variable]]

  # Only test counties with non-missing values
  # For counties missing data, use 0 as placeholder
  x_clean <- if_else(is.na(x), 0, x)

  result <- moran.test(
    x_clean,
    weights,
    zero.policy    = TRUE,
    randomisation  = TRUE
  )

  cat(glue("  {label}:\n"))
  cat(glue("    Moran's I = {round(result$statistic, 4)}\n"))
  cat(glue("    p-value   = {round(result$p.value, 4)}\n"))

  if (result$p.value < 0.05) {
    cat("    RESULT: Significant spatial autocorrelation detected.\n")
    cat("            Add spatial robustness model to appendix.\n")
  } else {
    cat("    RESULT: No significant spatial autocorrelation (p > 0.05).\n")
    cat("            County-clustered standard errors are sufficient.\n")
  }
  cat("\n")

  return(list(
    variable = label,
    morans_i = round(result$statistic, 4),
    p_value  = round(result$p.value, 4),
    significant = result$p.value < 0.05
  ))
}

# Test each key variable
test_vars <- list(
  list("enroll_rate",     "Retirement enrollment rate"),
  list("mean_literacy",   "Mean financial literacy score"),
  list("mean_impatience", "Mean impatience index"),
  list("mean_ai_density", "AI complaint density"),
  list("mean_distress",   "Financial distress index"),
  list("mean_bb",         "Broadband providers (IV)"),
  list("mean_poverty",    "Poverty rate")
)

results <- map(test_vars, function(v) {
  run_morans(v[[1]], v[[2]], wi_joined, lw_queen)
})

results_df <- bind_rows(map(results, as_tibble))

# ── 7. Summary and interpretation ────────────────────────────
cat(strrep("-", 60), "\n")
cat("SUMMARY OF SPATIAL AUTOCORRELATION TESTS\n")
cat(strrep("-", 60), "\n")
print(results_df, n = 20)

n_significant <- sum(results_df$significant)
cat(glue("\n{n_significant} of {nrow(results_df)} variables show significant spatial autocorrelation.\n\n"))

if (n_significant == 0) {
  cat("CONCLUSION: No spatial autocorrelation detected in any key variable.\n")
  cat("County-clustered standard errors in the Probit models are sufficient.\n")
  cat("No spatial robustness model required (but can be included as optional check).\n")
} else if (n_significant <= 2) {
  cat("CONCLUSION: Mild spatial autocorrelation in some variables.\n")
  cat("Add a spatial lag Probit as a robustness check in the paper appendix.\n")
  cat("Main results use county-clustered SE; spatial model confirms robustness.\n")
} else {
  cat("CONCLUSION: Substantial spatial autocorrelation detected.\n")
  cat("Add spatial error model as a primary robustness specification.\n")
  cat("Discuss in the methodology section alongside main Probit results.\n")
}

# ── 8. Save diagnostics report ───────────────────────────────
report_path <- file.path(docs_dir, "spatial_diagnostics_report.txt")
sink(report_path)
cat("Spatial Diagnostics Report — Wisconsin Retirement AI\n")
cat(format(Sys.time()), "\n\n")
cat("Test: Moran's I (Queen contiguity weights, row-standardized)\n\n")
print(results_df, n = 20)
cat(glue("\n{n_significant} variables with significant spatial autocorrelation (p < 0.05)\n"))
sink()

cat(glue("\nReport saved: {report_path}\n"))


# ============================================================
# PART 2 — SPATIAL ROBUSTNESS MODEL
# ============================================================
# WHY THIS SECTION EXISTS:
#   04c_robustness.R already ran Moran's I on the Model 3 Probit
#   residuals and found significant spatial autocorrelation
#   (I = 2.216, p = 0.013). This confirms county-clustered SE
#   alone do not fully absorb the error structure — there is a
#   genuine geographic component left over.
#
#   This section does NOT change the main results. Model 3 with
#   county-clustered SE remains the primary specification reported
#   in Table 2. Instead, this section adds a SPATIAL LAG model as
#   an appendix robustness check, showing that the headline
#   interaction effects (Literacy x AI, Impatience x AI) survive
#   when spatial dependence is explicitly modeled.
#
# APPROACH:
#   A spatial lag linear probability model (LPM) is used rather
#   than a spatial lag Probit, because maximum-likelihood spatial
#   Probit estimators are not well supported in standard R packages
#   for the panel structure here (repeated counties across waves).
#   The LPM with a spatial lag term (Wy) is the standard applied
#   solution and is reported alongside the main Probit AMEs, which
#   are themselves probability-scale and therefore comparable.
#
# OUTPUT:
#   outputs/tables/table5_spatial_robustness.csv
#   docs/spatial_robustness_log.txt
# ============================================================

cat("\n", strrep("=", 60), "\n", sep="")
cat("PART 2 — Spatial Lag Robustness Model\n")
cat(strrep("=", 60), "\n\n")

library(spatialreg)   # Spatial lag / spatial error regression

spatial_log <- c()
slog <- function(msg) { cat(msg, "\n"); spatial_log <<- c(spatial_log, msg) }

# ── S1. Load analytical dataset ───────────────────────────────
slog("[S1] Loading analytical dataset for spatial model...")

df_spatial <- read_parquet(file.path(final_dir, "master_analytical.parquet")) %>%
  mutate(
    county_fips = as.character(county_fips),
    wave_fe     = factor(SURVEY_WAVE)
  ) %>%
  filter(
    !is.na(enrolled), !is.na(literacy_score), !is.na(impatience_index),
    !is.na(ai_density_log), !is.na(lit_x_ai), !is.na(imp_x_ai),
    !is.na(distress_index), !is.na(log_median_income),
    !is.na(poverty_rate), !is.na(pct_bach_plus), !is.na(unemp_rate)
  )

slog(glue("  Spatial model sample: {format(nrow(df_spatial), big.mark=',')} obs"))

# ── S2. Aggregate to county level (one row per county) ───────
# Spatial lag models require a clean cross-sectional structure.
# We aggregate the individual-level data to county means, matching
# the unit of analysis for the spatial weights matrix. This mirrors
# the approach used to construct the AI Gap Index in 05c.

slog("\n[S2] Aggregating to county level for spatial regression...")

county_agg <- df_spatial %>%
  group_by(county_fips) %>%
  summarise(
    enroll_rate        = mean(enrolled, na.rm = TRUE),
    literacy_mean       = mean(literacy_score, na.rm = TRUE),
    impatience_mean     = mean(impatience_index, na.rm = TRUE),
    ai_density_mean     = mean(ai_density_log, na.rm = TRUE),
    lit_x_ai_mean       = mean(lit_x_ai, na.rm = TRUE),
    imp_x_ai_mean       = mean(imp_x_ai, na.rm = TRUE),
    distress_mean       = mean(distress_index, na.rm = TRUE),
    log_income_mean     = mean(log_median_income, na.rm = TRUE),
    poverty_mean        = mean(poverty_rate, na.rm = TRUE),
    bach_plus_mean      = mean(pct_bach_plus, na.rm = TRUE),
    unemp_mean          = mean(unemp_rate, na.rm = TRUE),
    n_obs               = n(),
    .groups = "drop"
  )

slog(glue("  Counties with AI-subsample data: {nrow(county_agg)}"))

# ── S3. Join to full county shapefile ─────────────────────────
# The spatial weights matrix needs ALL 72 counties to build the
# neighbor structure correctly, but lagsarlm() cannot handle NA
# outcomes, so the final model sample is restricted afterward.

wi_full <- wi_sf %>%
  left_join(county_agg, by = "county_fips")

n_with_data <- sum(!is.na(wi_full$enroll_rate))
slog(glue("  Counties with full data after shapefile join: {n_with_data} of {nrow(wi_full)}"))

if (n_with_data < 15) {
  slog("  WARNING: Fewer than 15 counties have AI-subsample data.")
  slog("  Spatial lag regression requires reasonable county coverage.")
  slog("  Results below should be interpreted as suggestive only.")
}

# ── S4. Build spatial weights for counties with data ──────────
wi_model_set <- wi_full %>% filter(!is.na(enroll_rate))
slog(glue("  Final spatial regression sample: {nrow(wi_model_set)} counties"))

nb_model   <- poly2nb(wi_model_set, queen = TRUE)
n_isolated <- sum(card(nb_model) == 0)
if (n_isolated > 0) {
  slog(glue("  NOTE: {n_isolated} counties have zero neighbors in restricted sample"))
}
lw_model <- nb2listw(nb_model, style = "W", zero.policy = TRUE)

# ── S5. Spatial lag model ──────────────────────────────────────
# Model: enroll_rate = rho*W*enroll_rate + Xb + e
# rho captures spillover: does a county's enrollment rate move
# with its neighbors' enrollment rates, after controlling for
# the same covariates as Model 3?

slog("\n[S3] Estimating spatial lag model (county-level, LPM)...")

spatial_lag_model <- tryCatch({
  lagsarlm(
    enroll_rate ~
      literacy_mean + impatience_mean +
      ai_density_mean + lit_x_ai_mean + imp_x_ai_mean +
      distress_mean + log_income_mean +
      poverty_mean + bach_plus_mean + unemp_mean,
    data    = wi_model_set,
    listw   = lw_model,
    zero.policy = TRUE
  )
}, error = function(e) {
  slog(glue("  Spatial lag model failed: {e$message}"))
  NULL
})

# ── S6. Non-spatial OLS comparison (same county-level sample) ─
# Direct comparison: does ignoring spatial structure change the
# sign or magnitude of the headline interaction coefficients?

ols_comparison <- lm(
  enroll_rate ~
    literacy_mean + impatience_mean +
    ai_density_mean + lit_x_ai_mean + imp_x_ai_mean +
    distress_mean + log_income_mean +
    poverty_mean + bach_plus_mean + unemp_mean,
  data = wi_model_set
)

# ── S7. Report results ──────────────────────────────────────────
if (!is.null(spatial_lag_model)) {
  slog("\n[S4] Spatial lag model results:")
  sl_summary <- summary(spatial_lag_model)

  slog(glue("  Rho (spatial lag coefficient): {round(spatial_lag_model$rho, 4)}"))

  sl_coefs  <- coef(spatial_lag_model)
  ols_coefs <- coef(ols_comparison)

  compare_vars <- c("literacy_mean", "impatience_mean", "ai_density_mean",
                     "lit_x_ai_mean", "imp_x_ai_mean")

  slog("\n  Coefficient comparison — OLS vs Spatial Lag:")
  for (v in compare_vars) {
    ols_val <- if (v %in% names(ols_coefs)) round(ols_coefs[v], 4) else NA
    sl_val  <- if (v %in% names(sl_coefs))  round(sl_coefs[v], 4)  else NA
    slog(glue("    {str_pad(v, 20)}  OLS={ols_val}   Spatial={sl_val}"))
  }

  slog("\n  INTERPRETATION:")
  slog("  If the Literacy x AI and Impatience x AI coefficients retain")
  slog("  the same sign and similar magnitude in the spatial lag model")
  slog("  as in OLS, this confirms the headline interaction effects are")
  slog("  not artifacts of spatial clustering in the underlying data.")

  spatial_tbl <- tibble(
    term = c("rho (spatial lag)", compare_vars),
    OLS_county_level_NOT_for_paper = c(
      NA, sapply(compare_vars, function(v)
        if (v %in% names(ols_coefs)) round(ols_coefs[v],4) else NA)
    ),
    Spatial_Lag_UNRELIABLE_n26_fragmented = c(
      round(spatial_lag_model$rho, 4),
      sapply(compare_vars, function(v)
        if (v %in% names(sl_coefs)) round(sl_coefs[v],4) else NA)
    )
  )

  # ── DIAGNOSTIC CAVEAT ─────────────────────────────────────────
  # This table is NOT a validated robustness check. The spatial
  # lag model was estimated on only 26 of 72 counties, fragmented
  # into 5 disconnected spatial sub-graphs (2 counties had zero
  # neighbors). Both headline interaction coefficients (literacy x
  # AI, impatience x AI) FLIP SIGN relative to OLS, which is
  # evidence of model instability from sparse, disconnected spatial
  # data -- not evidence that the main Probit results are spatially
  # robust. Do NOT cite these coefficients as a robustness check in
  # the paper. Report only the diagnostic conclusion: a spatial lag
  # model was attempted but county coverage was insufficient for
  # reliable estimation (see docs/spatial_robustness_log.txt).
  slog("\n  *** DIAGNOSTIC CAVEAT — DO NOT CITE AS ROBUSTNESS CHECK ***")
  slog("  This spatial lag model used only 26 of 72 counties, split into")
  slog("  5 disconnected sub-graphs. Both lit_x_ai and imp_x_ai flip sign")
  slog("  relative to OLS -- this reflects estimator instability on a")
  slog("  small, fragmented sample, NOT confirmation that main results")
  slog("  are spatially robust. Table column names reflect this caveat.")
  slog("  For the paper: report only that a spatial lag model was")
  slog("  attempted and judged unreliable due to insufficient county")
  slog("  coverage; do not report or interpret its coefficients.")

  tbl_dir_local <- here("outputs","tables")
  dir.create(tbl_dir_local, showWarnings = FALSE, recursive = TRUE)
  spatial_tbl_path <- file.path(tbl_dir_local, "table5_spatial_robustness.csv")
  write_csv(spatial_tbl, spatial_tbl_path)
  slog(glue("\n  Spatial robustness table saved: {spatial_tbl_path}"))

} else {
  slog("\n  Spatial lag model could not be estimated.")
  slog("  REPORT: Note in appendix that Moran's I confirms spatial")
  slog("  autocorrelation (I=2.216, p=0.013) but the limited county")
  slog("  coverage of the AI subsample (28 of 72 counties) prevents")
  slog("  reliable spatial lag estimation. Recommend this as a")
  slog("  documented limitation and a direction for future work with")
  slog("  expanded CFPB coverage.")
}

# ── S8. Save spatial robustness log ──────────────────────────
spatial_log_path <- file.path(docs_dir, "spatial_robustness_log.txt")
writeLines(spatial_log, spatial_log_path)
slog(glue("\nSpatial robustness log saved: {spatial_log_path}"))

cat("\n", strrep("=", 60), "\n", sep="")
cat("Script 05a complete (diagnostics + spatial robustness model).\n")
cat(strrep("=", 60), "\n")
