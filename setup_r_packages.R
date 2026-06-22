# ============================================================
# Wisconsin Retirement AI — R Package Installation
# SIMPLIFIED VERSION (no renv — install directly)
# ============================================================
# HOW TO RUN:
#   1. Open RStudio
#   2. Open this file
#   3. Select ALL (Ctrl+A) and click Run
#      OR paste into the RStudio Console and press Enter
#
# This installs all required packages into your main R library.
# Takes approximately 5-15 minutes on first run.
# Packages only need to be installed once.
#
# After this runs successfully, renv can be added later for
# reproducibility locking — but it is not needed to start analysis.
# ============================================================

# ── Set CRAN mirror explicitly (avoids "no mirror selected" error) ──
options(repos = c(CRAN = "https://cloud.r-project.org"))

# ── Helper: install only if not already installed ────────────
install_if_missing <- function(pkgs) {
  to_install <- pkgs[!pkgs %in% rownames(installed.packages())]
  if (length(to_install) == 0) {
    message("All packages already installed.")
    return(invisible(NULL))
  }
  message(paste("Installing:", paste(to_install, collapse = ", ")))
  install.packages(to_install, dependencies = TRUE)
}

# ── Package list ─────────────────────────────────────────────
packages <- c(
  # Data wrangling
  "tidyverse",
  "haven",        # Read SPSS .sav and Stata .dta files (NFCS waves)
  "labelled",     # Work with labelled SPSS variables
  "janitor",      # Clean column names
  "data.table",   # Fast large-file operations
  "arrow",        # Read/write parquet files

  # Geographic / Census
  "tidycensus",   # Census API — ACS county data
  "tigris",       # Census TIGER shapefiles
  "sf",           # Spatial data handling
  "units",        # Units for spatial operations

  # Econometrics
  "fixest",       # Fast fixed effects + clustered SE
  "margins",      # Average marginal effects from Probit
  "sampleSelection", # IV-Probit / Heckman
  "lmtest",       # Coefficient tests
  "sandwich",     # Robust / clustered standard errors
  "pwr",          # Power analysis
  "AER",          # Applied Econometrics — ivreg()

  # Publication output
  "modelsummary", # Regression tables
  "gt",           # Table formatting
  "kableExtra",   # LaTeX/HTML table export
  "stargazer",    # Alternative regression tables

  # Visualization
  "ggeffects",    # Predicted probability plots
  "patchwork",    # Combine multiple ggplots
  "scales",       # Axis formatting
  "viridis",      # Colorblind-safe palettes
  "RColorBrewer", # Color palettes

  # Utilities
  "here",         # Relative file paths
  "glue",         # String interpolation
  "lubridate",    # Date handling
  "readxl",       # Read Excel files (HUD crosswalk)
  "writexl",      # Write Excel files
  "tictoc"        # Timing code blocks
)

# ── Install ───────────────────────────────────────────────────
message("\n============================================")
message("Installing R packages for Wisconsin Retirement AI")
message("This may take 5-15 minutes...")
message("============================================\n")

install_if_missing(packages)

# ── Verify installation ───────────────────────────────────────
message("\n============================================")
message("Verifying installations...")
message("============================================\n")

failed <- c()
for (pkg in packages) {
  ok <- requireNamespace(pkg, quietly = TRUE)
  status <- if (ok) "OK" else "FAILED"
  message(sprintf("  %-20s %s", pkg, status))
  if (!ok) failed <- c(failed, pkg)
}

if (length(failed) == 0) {
  message("\n✓ All packages installed successfully.")
  message("✓ You are ready to run the analysis scripts.")
} else {
  message(paste("\n✗ Failed to install:", paste(failed, collapse = ", ")))
  message("Try installing failed packages manually:")
  message(paste0('install.packages(c("', paste(failed, collapse='", "'), '"))'))
}

# ── Quick test: load the most critical packages ───────────────
message("\nQuick load test for critical packages...")

critical <- c("tidyverse", "haven", "tidycensus", "arrow",
              "fixest", "modelsummary", "sf", "here", "readxl")

for (pkg in critical) {
  tryCatch({
    library(pkg, character.only = TRUE, quietly = TRUE)
    message(sprintf("  %-20s loaded OK", pkg))
  }, error = function(e) {
    message(sprintf("  %-20s LOAD ERROR: %s", pkg, e$message))
  })
}

message("\n============================================")
message("Setup complete.")
message("Next step: Set your Census API key:")
message('  tidycensus::census_api_key("YOUR_KEY_HERE", install = TRUE)')
message("Get a free key at: https://api.census.gov/data/key_signup.html")
message("============================================\n")
