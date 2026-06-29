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

cat(glue("  Average neighbors per county: {mean(card(nb_queen)):.1f}\n"))
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

cat("\n", strrep("=", 60), "\n", sep="")
cat("Script 05a complete.\n")
cat(strrep("=", 60), "\n")
