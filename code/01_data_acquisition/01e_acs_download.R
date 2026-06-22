# ============================================================
# Workstream 1E: American Community Survey — REVISED v3
# County-Level Controls for Wisconsin (All 72 Counties)
# ============================================================
# Project: Wisconsin Retirement AI
# Script:  code/01_data_acquisition/01e_acs_download.R
# Author:  Khawaja Sazzad Ali
# Date:    Summer 2026
#
# FIXES FROM v2:
#   - Switched from output="wide" to output="tidy" (long format)
#     to avoid column rename collisions entirely
#   - Downloads variables one group at a time, tests availability
#     before requesting, so missing 2009 vars fail gracefully
#   - Pivots to wide format AFTER renaming, which is clean and safe
#   - No more rename vector inversion bug
#
# HOW TO RUN:
#   1. Make sure your Census API key is set:
#      census_api_key("YOUR_KEY", install=TRUE)
#   2. Restart RStudio after setting key
#   3. Open this file, Ctrl+A, click Run
# ============================================================

library(tidyverse)
library(tidycensus)
library(arrow)
library(here)

# ── 0. Paths ─────────────────────────────────────────────────
proc_dir <- here("data", "processed")
docs_dir <- here("docs")
dir.create(proc_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(docs_dir, showWarnings = FALSE, recursive = TRUE)

# ── 1. Settings ───────────────────────────────────────────────
ACS_YEARS      <- c(2009, 2012, 2015, 2018, 2021, 2022)
WISCONSIN_FIPS <- "55"

NFCS_WAVE_MAP <- c(
  "2009" = 2009, "2012" = 2012, "2015" = 2015,
  "2018" = 2018, "2021" = 2021, "2022" = 2024
)

# ── 2. Variable definitions ───────────────────────────────────
# Each entry: Census_variable_code = "our_name"
# Grouped so we can skip unavailable groups gracefully

VARS_POPULATION <- c(
  "B01003_001" = "pop_total"
)

VARS_INCOME <- c(
  "B19013_001" = "median_hh_income"
)

VARS_POVERTY <- c(
  "B17001_002" = "pop_below_poverty",
  "B17001_001" = "pop_poverty_denom"
)

# B15003 not available in 2009 — handled by fallback below
VARS_EDUCATION <- c(
  "B15003_001" = "pop_edu_total",
  "B15003_022" = "pop_bach_degree",
  "B15003_023" = "pop_masters",
  "B15003_024" = "pop_professional",
  "B15003_025" = "pop_doctorate"
)

# Fallback education for 2009 (uses B15002 which exists in 2009)
VARS_EDUCATION_2009 <- c(
  "B15002_001" = "pop_edu_total",    # Total population 25+
  "B15002_015" = "pop_bach_degree",  # Male: Bachelor's
  "B15002_016" = "pop_masters",      # Male: Master's
  "B15002_017" = "pop_professional", # Male: Professional
  "B15002_018" = "pop_doctorate",    # Male: Doctorate
  "B15002_032" = "pop_bach_f",       # Female: Bachelor's
  "B15002_033" = "pop_masters_f",    # Female: Master's
  "B15002_034" = "pop_professional_f",
  "B15002_035" = "pop_doctorate_f"
)

VARS_RACE <- c(
  "B03002_001" = "pop_race_total",
  "B03002_003" = "pop_white_nh",
  "B03002_004" = "pop_black_nh",
  "B03002_012" = "pop_hispanic",
  "B03002_006" = "pop_asian_nh"
)

VARS_HOUSING <- c(
  "B25003_001" = "pop_tenure_total",
  "B25003_002" = "pop_owner_occ"
)

# Broadband only from 2016 onward
VARS_BROADBAND <- c(
  "B28002_001" = "pop_internet_total",
  "B28002_004" = "pop_broadband_sub"
)


# ── 3. Core download helper ───────────────────────────────────
# Downloads one group of variables for one year.
# Returns a tidy (long) data frame or NULL if unavailable.

fetch_vars <- function(var_group, year, state_fips) {
  tryCatch({
    df <- get_acs(
      geography = "county",
      variables = names(var_group),  # Census codes
      state     = state_fips,
      year      = year,
      survey    = "acs5",
      output    = "tidy"             # Long format: one row per county-variable
    )
    # In tidy output: columns are GEOID, NAME, variable, estimate, moe
    # 'variable' column contains the Census code (e.g. "B01003_001")
    # Replace Census codes with our names using the lookup vector
    df$variable <- var_group[df$variable]
    # Keep only rows where variable was successfully mapped
    df <- df[!is.na(df$variable), ]
    return(df)
  }, error = function(e) {
    # Silently return NULL — caller handles missing groups
    return(NULL)
  })
}


# ── 4. Main per-year download function ────────────────────────
download_acs_year <- function(year) {
  
  cat(sprintf("\n--- ACS %d ---\n", year))
  
  results <- list()
  
  # Population
  r <- fetch_vars(VARS_POPULATION, year, WISCONSIN_FIPS)
  if (!is.null(r)) { results[["pop"]] <- r; cat("  population: OK\n") }
  
  # Income
  r <- fetch_vars(VARS_INCOME, year, WISCONSIN_FIPS)
  if (!is.null(r)) { results[["inc"]] <- r; cat("  income: OK\n") }
  
  # Poverty
  r <- fetch_vars(VARS_POVERTY, year, WISCONSIN_FIPS)
  if (!is.null(r)) { results[["pov"]] <- r; cat("  poverty: OK\n") }
  
  # Education — try main vars, fall back to 2009 version
  r <- fetch_vars(VARS_EDUCATION, year, WISCONSIN_FIPS)
  if (!is.null(r)) {
    results[["edu"]] <- r
    cat("  education (B15003): OK\n")
  } else {
    r2 <- fetch_vars(VARS_EDUCATION_2009, year, WISCONSIN_FIPS)
    if (!is.null(r2)) {
      # For 2009: combine male + female bachelor's+ into single variables
      r2 <- r2 %>%
        group_by(GEOID, NAME) %>%
        summarise(
          pop_edu_total  = estimate[variable == "pop_edu_total"],
          pop_bach_degree = sum(estimate[variable %in%
                                           c("pop_bach_degree", "pop_bach_f")], na.rm = TRUE),
          pop_masters    = sum(estimate[variable %in%
                                          c("pop_masters", "pop_masters_f")], na.rm = TRUE),
          pop_professional = sum(estimate[variable %in%
                                            c("pop_professional", "pop_professional_f")], na.rm = TRUE),
          pop_doctorate  = sum(estimate[variable %in%
                                          c("pop_doctorate", "pop_doctorate_f")], na.rm = TRUE),
          .groups = "drop"
        ) %>%
        pivot_longer(
          cols      = -c(GEOID, NAME),
          names_to  = "variable",
          values_to = "estimate"
        ) %>%
        mutate(moe = NA_real_)
      results[["edu"]] <- r2
      cat("  education (B15002 fallback): OK\n")
    } else {
      cat("  education: NOT AVAILABLE for this year\n")
    }
  }
  
  # Race/ethnicity
  r <- fetch_vars(VARS_RACE, year, WISCONSIN_FIPS)
  if (!is.null(r)) { results[["race"]] <- r; cat("  race/ethnicity: OK\n") }
  
  # Housing tenure
  r <- fetch_vars(VARS_HOUSING, year, WISCONSIN_FIPS)
  if (!is.null(r)) { results[["hous"]] <- r; cat("  housing: OK\n") }
  
  # Broadband (2016+ only)
  if (year >= 2016) {
    r <- fetch_vars(VARS_BROADBAND, year, WISCONSIN_FIPS)
    if (!is.null(r)) { results[["bb"]] <- r; cat("  broadband: OK\n") }
  }
  
  if (length(results) == 0) {
    cat("  No variables downloaded for this year.\n")
    return(NULL)
  }
  
  # ── Combine all variable groups ───────────────────────────
  df_long <- bind_rows(results)
  
  # ── Pivot to wide: one row per county, one col per variable ─
  df_wide <- df_long %>%
    select(GEOID, NAME, variable, estimate) %>%
    pivot_wider(
      names_from  = variable,
      values_from = estimate,
      values_fn   = mean   # If duplicates exist, take mean
    ) %>%
    rename(
      county_fips = GEOID,
      county_name = NAME
    ) %>%
    mutate(
      acs_year   = year,
      nfcs_wave  = as.integer(NFCS_WAVE_MAP[as.character(year)]),
      state_fips = WISCONSIN_FIPS
    )
  
  cat(sprintf("  Result: %d counties, %d variables\n",
              nrow(df_wide), ncol(df_wide)))
  return(df_wide)
}


# ── 5. Run all years ──────────────────────────────────────────
cat(strrep("=", 60), "\n")
cat("Workstream 1E: ACS County Data\n")
cat(strrep("=", 60), "\n")

all_acs <- map(ACS_YEARS, download_acs_year)
all_acs <- compact(all_acs)

if (length(all_acs) == 0) {
  stop(paste(
    "No ACS data downloaded.",
    "Check Census API key: census_api_key('YOUR_KEY', install=TRUE)",
    "Then restart RStudio and re-run.",
    sep = "\n"
  ))
}

cat(sprintf("\nDownloaded %d of %d years successfully.\n",
            length(all_acs), length(ACS_YEARS)))


# ── 6. Combine years ──────────────────────────────────────────
df_acs <- bind_rows(all_acs)
cat(sprintf("Combined: %d rows x %d columns\n", nrow(df_acs), ncol(df_acs)))

# Convert all estimate columns to numeric
df_acs <- df_acs %>%
  mutate(across(
    -c(county_fips, county_name, state_fips, acs_year, nfcs_wave),
    ~ suppressWarnings(as.numeric(.x))
  ))


# ── 7. Derived variables ──────────────────────────────────────
df_acs <- df_acs %>%
  mutate(
    poverty_rate = if_else(
      !is.na(pop_poverty_denom) & pop_poverty_denom > 0,
      pop_below_poverty / pop_poverty_denom * 100, NA_real_
    ),
    pct_bach_plus = if_else(
      !is.na(pop_edu_total) & pop_edu_total > 0,
      (coalesce(pop_bach_degree, 0) + coalesce(pop_masters, 0) +
         coalesce(pop_professional, 0) + coalesce(pop_doctorate, 0)) /
        pop_edu_total * 100, NA_real_
    ),
    pct_white_nh = if_else(
      !is.na(pop_race_total) & pop_race_total > 0,
      coalesce(pop_white_nh, 0) / pop_race_total * 100, NA_real_
    ),
    pct_black_nh = if_else(
      !is.na(pop_race_total) & pop_race_total > 0,
      coalesce(pop_black_nh, 0) / pop_race_total * 100, NA_real_
    ),
    pct_hispanic = if_else(
      !is.na(pop_race_total) & pop_race_total > 0,
      coalesce(pop_hispanic, 0) / pop_race_total * 100, NA_real_
    ),
    pct_asian_nh = if_else(
      !is.na(pop_race_total) & pop_race_total > 0,
      coalesce(pop_asian_nh, 0) / pop_race_total * 100, NA_real_
    ),
    pct_owner_occ = if_else(
      !is.na(pop_tenure_total) & pop_tenure_total > 0,
      coalesce(pop_owner_occ, 0) / pop_tenure_total * 100, NA_real_
    ),
    pct_broadband = if_else(
      !is.na(pop_internet_total) & pop_internet_total > 0,
      coalesce(pop_broadband_sub, 0) / pop_internet_total * 100, NA_real_
    ),
    log_median_income = log(coalesce(median_hh_income, NA_real_))
  )


# ── 8. Final column selection ─────────────────────────────────
final_cols <- c(
  "county_fips", "county_name", "state_fips", "acs_year", "nfcs_wave",
  "pop_total", "median_hh_income", "log_median_income",
  "poverty_rate", "pct_bach_plus",
  "pct_white_nh", "pct_black_nh", "pct_hispanic", "pct_asian_nh",
  "pct_owner_occ", "pct_broadband"
)
final_cols <- final_cols[final_cols %in% names(df_acs)]
df_final   <- df_acs[, final_cols] %>% arrange(county_fips, acs_year)

cat(sprintf("\nFinal dataset: %d rows x %d columns\n",
            nrow(df_final), ncol(df_final)))

# Missingness summary
cat("\nMissing values per variable:\n")
miss <- colSums(is.na(df_final))
miss <- miss[miss > 0]
if (length(miss) == 0) {
  cat("  None.\n")
} else {
  for (nm in names(miss)) {
    cat(sprintf("  %-30s %d (%.1f%%)\n",
                nm, miss[[nm]], miss[[nm]] / nrow(df_final) * 100))
  }
}

# Quick sanity check
cat("\nSample — Wisconsin county statistics (2021):\n")
sample_2021 <- df_final %>%
  filter(acs_year == 2021) %>%
  summarise(
    counties       = n(),
    mean_poverty   = round(mean(poverty_rate,    na.rm=TRUE), 1),
    mean_bach_plus = round(mean(pct_bach_plus,   na.rm=TRUE), 1),
    mean_black_nh  = round(mean(pct_black_nh,    na.rm=TRUE), 1),
    mean_hispanic  = round(mean(pct_hispanic,    na.rm=TRUE), 1)
  )
print(sample_2021)


# ── 9. Save ───────────────────────────────────────────────────
out_path <- file.path(proc_dir, "acs_county_controls.parquet")
write_parquet(df_final, out_path)
cat(sprintf("\nSaved: %s\n", out_path))

cat("\n", strrep("=", 60), "\n", sep="")
cat("Workstream 1E complete.\n")
cat(strrep("=", 60), "\n", sep="")