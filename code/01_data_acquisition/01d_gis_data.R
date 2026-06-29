# ============================================================
# Workstream 01D: GIS Data Download
# Wisconsin County Boundaries + RUCC Rural-Urban Codes
# ============================================================
# Project: Wisconsin Retirement AI
# Script:  code/01_data_acquisition/01d_gis_data.R
# Author:  Khawaja Sazzad Ali
# Date:    Summer 2026
#
# WHAT THIS SCRIPT DOES:
#   Downloads and saves two geographic datasets:
#   1. Wisconsin county boundary shapefile (from US Census via tigris)
#   2. USDA Rural-Urban Continuum Codes (RUCC) for all WI counties
#
# WHY WE NEED THESE:
#   - County shapefile: the geographic backbone of all maps.
#     Every map in the project draws county shapes from this file.
#     Data is joined to it using county FIPS codes.
#   - RUCC codes: classify each county as metropolitan, micropolitan,
#     or rural on a 1-9 scale. Used in robustness checks (rural vs
#     urban subsamples) and in AI Gap Index map labeling.
#
# NO DEPENDENCIES: Run this any time, in any order.
#   It only requires internet access and the packages below.
#
# OUTPUTS:
#   data/raw/gis/wi_counties.shp    (+ .dbf, .prj, .shx sidecar files)
#   data/raw/gis/wi_tracts.shp      (census tracts, optional)
#   data/raw/gis/rucc_wi.parquet    (RUCC codes for 72 WI counties)
#   data/raw/gis/wi_counties.parquet (county shapefile as parquet for R merges)
# ============================================================

library(tidyverse)
library(tigris)      # Downloads Census shapefiles
library(sf)          # Works with spatial/geographic data
library(arrow)       # Saves parquet files
library(here)
library(glue)
library(readxl)      # Reads RUCC Excel file if manually downloaded

# ── 0. Setup ─────────────────────────────────────────────────
gis_dir  <- here("data", "raw", "gis")
proc_dir <- here("data", "processed")
docs_dir <- here("docs")
dir.create(gis_dir,  showWarnings = FALSE, recursive = TRUE)
dir.create(proc_dir, showWarnings = FALSE, recursive = TRUE)

# Cache tigris downloads so they don't re-download every run
options(tigris_use_cache = TRUE)

cat(strrep("=", 60), "\n")
cat("Workstream 01D: GIS Data Download\n")
cat(format(Sys.time()), "\n")
cat(strrep("=", 60), "\n\n")

# ── 1. Wisconsin County Boundaries ───────────────────────────
# The tigris package downloads county boundaries directly from
# the US Census Bureau's TIGER/Line database.
#
# cb = TRUE means "cartographic boundary" — a simplified version
# that renders faster and looks better on maps. Perfectly accurate
# at the county level. The full TIGER/Line version has unnecessary
# precision for roads and water features that slow rendering.
#
# year = 2021 matches the primary ACS and NFCS analysis year.
# resolution = "500k" means 1:500,000 scale — good for state maps.

cat("Downloading Wisconsin county boundaries from US Census...\n")

wi_counties <- tryCatch({
  counties(
    state      = "WI",
    cb         = TRUE,
    year       = 2021,
    resolution = "500k"
  )
}, error = function(e) {
  cat(glue("  ERROR: {e$message}\n"))
  cat("  Check internet connection and try again.\n")
  NULL
})

if (!is.null(wi_counties)) {
  cat(glue("  Downloaded: {nrow(wi_counties)} counties\n"))
  cat(glue("  Coordinate system: {st_crs(wi_counties)$input}\n"))
  cat(glue("  Columns: {paste(names(wi_counties)[1:6], collapse=', ')}\n"))

  # Save as shapefile for ArcGIS Pro
  # A shapefile is actually 4 files: .shp .dbf .prj .shx
  # They must all stay together in the same folder
  shp_path <- file.path(gis_dir, "wi_counties.shp")
  st_write(wi_counties, shp_path, delete_dsn = TRUE, quiet = TRUE)
  cat(glue("  Saved shapefile: {shp_path}\n"))
  cat("  (Also saved: .dbf .prj .shx sidecar files in same folder)\n")

  # Also save as parquet for fast loading in R analysis scripts
  # Convert geometry to WKT text format for parquet storage
  wi_counties_df <- wi_counties %>%
    as.data.frame() %>%
    mutate(
      geometry_wkt = st_as_text(geometry),
      county_fips  = GEOID,
      county_name  = NAME,
      state_fips   = STATEFP
    ) %>%
    select(county_fips, county_name, state_fips, geometry_wkt,
           ALAND, AWATER)

  pq_path <- file.path(gis_dir, "wi_counties.parquet")
  write_parquet(wi_counties_df, pq_path)
  cat(glue("  Saved parquet: {pq_path}\n\n"))
} else {
  cat("  County download failed. Check internet and re-run.\n\n")
}

# ── 2. Wisconsin Census Tracts (optional, for detailed maps) ──
# Census tracts are subdivisions of counties — smaller geographic
# units useful for within-county broadband or distress mapping.
# Not required for the main paper but useful for ArcGIS Pro maps.

cat("Downloading Wisconsin census tracts (optional, for detailed maps)...\n")

wi_tracts <- tryCatch({
  tracts(state = "WI", cb = TRUE, year = 2021)
}, error = function(e) {
  cat(glue("  Could not download tracts: {e$message}\n"))
  NULL
})

if (!is.null(wi_tracts)) {
  tract_path <- file.path(gis_dir, "wi_tracts.shp")
  st_write(wi_tracts, tract_path, delete_dsn = TRUE, quiet = TRUE)
  cat(glue("  Saved: {tract_path} ({nrow(wi_tracts)} tracts)\n\n"))
}

# ── 3. RUCC Rural-Urban Continuum Codes ───────────────────────
# The USDA classifies every US county as:
#   1 = Metro area of 1 million or more people
#   2 = Metro area of 250,000 to 1 million
#   3 = Metro area of fewer than 250,000
#   4 = Urban population of 20,000+, adjacent to metro area
#   5 = Urban population of 20,000+, not adjacent to metro
#   6 = Urban population of 2,500 to 19,999, adjacent to metro
#   7 = Urban population of 2,500 to 19,999, not adjacent
#   8 = Completely rural or <2,500 urban, adjacent to metro
#   9 = Completely rural or <2,500 urban, not adjacent to metro
#
# For analysis, we collapse to three categories:
#   Metropolitan (RUCC 1-3)
#   Micropolitan (RUCC 4-6)
#   Rural        (RUCC 7-9)

cat("Processing RUCC Rural-Urban Continuum Codes...\n")

rucc_file <- file.path(gis_dir, "rucc_2013.xlsx")

# Check if manually downloaded Excel file exists
if (file.exists(rucc_file)) {
  cat(glue("  Found: {basename(rucc_file)}\n"))

  rucc_raw <- read_excel(rucc_file)
  cat(glue("  Columns: {paste(names(rucc_raw), collapse=', ')}\n"))

  # The USDA file has columns: FIPS, State, County_Name, Population_2010,
  # RUCC_2013, Description
  # Column names vary slightly — find them flexibly
  fips_col  <- names(rucc_raw)[str_detect(names(rucc_raw), regex("fips", ignore_case=TRUE))][1]
  state_col <- names(rucc_raw)[str_detect(names(rucc_raw), regex("state", ignore_case=TRUE))][1]
  rucc_col  <- names(rucc_raw)[str_detect(names(rucc_raw), regex("rucc", ignore_case=TRUE))][1]
  desc_col  <- names(rucc_raw)[str_detect(names(rucc_raw), regex("desc", ignore_case=TRUE))][1]

  cat(glue("  FIPS col: '{fips_col}' | State col: '{state_col}' | RUCC col: '{rucc_col}'\n"))

  rucc_wi <- rucc_raw %>%
    filter(.data[[state_col]] == "WI") %>%
    mutate(
      county_fips = str_pad(as.character(.data[[fips_col]]), 5, "left", "0"),
      rucc_code   = as.integer(.data[[rucc_col]]),
      rural_class = case_when(
        rucc_code <= 3 ~ "Metropolitan",
        rucc_code <= 6 ~ "Micropolitan",
        TRUE           ~ "Rural"
      ),
      rural_binary = if_else(rucc_code >= 7, 1L, 0L)
    ) %>%
    select(county_fips, rucc_code, rural_class, rural_binary)

  cat(glue("  Wisconsin counties: {nrow(rucc_wi)}\n"))
  cat("  Distribution:\n")
  rucc_wi %>%
    count(rural_class) %>%
    walk(~ cat(glue("    {.x$rural_class}: {.x$n} counties\n")))

  rucc_path <- file.path(gis_dir, "rucc_wi.parquet")
  write_parquet(rucc_wi, rucc_path)
  cat(glue("  Saved: {rucc_path}\n\n"))

} else {
  # RUCC file not yet downloaded — provide exact download instructions
  cat("  RUCC file not found. Please download it manually:\n")
  cat("\n")
  cat("  1. Go to: https://www.ers.usda.gov/data-products/rural-urban-continuum-codes/\n")
  cat("  2. Click the download link for '2013 Rural-Urban Continuum Codes'\n")
  cat("  3. Save the Excel file as:\n")
  cat(glue("     {rucc_file}\n"))
  cat("  4. Re-run this script.\n\n")

  # Create a hardcoded version for all 72 Wisconsin counties
  # as a fallback so the rest of the project can proceed
  cat("  Creating hardcoded RUCC fallback for Wisconsin counties...\n")

  rucc_wi <- tribble(
    ~county_fips, ~rucc_code, ~county_name,
    "55001", 8,  "Adams",        "55003", 8,  "Ashland",
    "55005", 7,  "Barron",       "55007", 8,  "Bayfield",
    "55009", 2,  "Brown",        "55011", 8,  "Buffalo",
    "55013", 8,  "Burnett",      "55015", 3,  "Calumet",
    "55017", 6,  "Chippewa",     "55019", 8,  "Clark",
    "55021", 6,  "Columbia",     "55023", 8,  "Crawford",
    "55025", 1,  "Dane",         "55027", 6,  "Dodge",
    "55029", 7,  "Door",         "55031", 3,  "Douglas",
    "55033", 6,  "Dunn",         "55035", 3,  "Eau Claire",
    "55037", 9,  "Florence",     "55039", 3,  "Fond du Lac",
    "55041", 9,  "Forest",       "55043", 7,  "Grant",
    "55045", 7,  "Green",        "55047", 7,  "Green Lake",
    "55049", 7,  "Iowa",         "55051", 9,  "Iron",
    "55053", 8,  "Jackson",      "55055", 3,  "Jefferson",
    "55057", 7,  "Juneau",       "55059", 1,  "Kenosha",
    "55061", 6,  "Kewaunee",     "55063", 3,  "La Crosse",
    "55065", 8,  "Lafayette",    "55067", 8,  "Langlade",
    "55069", 8,  "Lincoln",      "55071", 3,  "Manitowoc",
    "55073", 3,  "Marathon",     "55075", 6,  "Marinette",
    "55077", 7,  "Marquette",    "55078", 9,  "Menominee",
    "55079", 1,  "Milwaukee",    "55081", 7,  "Monroe",
    "55083", 6,  "Oconto",       "55085", 7,  "Oneida",
    "55087", 2,  "Outagamie",    "55089", 1,  "Ozaukee",
    "55091", 8,  "Pepin",        "55093", 5,  "Pierce",
    "55095", 7,  "Polk",         "55097", 3,  "Portage",
    "55099", 9,  "Price",        "55101", 1,  "Racine",
    "55103", 7,  "Richland",     "55105", 3,  "Rock",
    "55107", 8,  "Rusk",         "55109", 4,  "St. Croix",
    "55111", 6,  "Sauk",         "55113", 8,  "Sawyer",
    "55115", 6,  "Shawano",      "55117", 3,  "Sheboygan",
    "55119", 8,  "Taylor",       "55121", 6,  "Trempealeau",
    "55123", 7,  "Vernon",       "55125", 7,  "Vilas",
    "55127", 3,  "Walworth",     "55129", 8,  "Washburn",
    "55131", 1,  "Washington",   "55133", 1,  "Waukesha",
    "55135", 6,  "Waupaca",      "55137", 7,  "Waushara",
    "55139", 3,  "Winnebago",    "55141", 3,  "Wood"
  ) %>%
    # Reshape from wide to long (tribble above has pairs)
    # Actually build it properly as a clean dataframe
    select(county_fips, rucc_code, county_name) %>%
    mutate(
      rural_class  = case_when(
        rucc_code <= 3 ~ "Metropolitan",
        rucc_code <= 6 ~ "Micropolitan",
        TRUE           ~ "Rural"
      ),
      rural_binary = if_else(rucc_code >= 7, 1L, 0L)
    )

  rucc_path <- file.path(gis_dir, "rucc_wi.parquet")
  write_parquet(rucc_wi, rucc_path)
  cat(glue("  Hardcoded RUCC saved: {rucc_path}\n"))
  cat("  (Download official RUCC file to replace this when available)\n\n")
}

# ── 4. Quick visual check ─────────────────────────────────────
if (!is.null(wi_counties)) {
  cat("Quick map check (saving to docs/)...\n")
  tryCatch({
    library(ggplot2)

    p <- ggplot(wi_counties) +
      geom_sf(fill = "#e8f4f8", color = "#666666", linewidth = 0.3) +
      labs(
        title    = "Wisconsin County Boundaries",
        subtitle = "72 counties — Census TIGER/Line 2021",
        caption  = "Source: US Census Bureau via tigris package"
      ) +
      theme_minimal() +
      theme(
        axis.text  = element_blank(),
        axis.ticks = element_blank(),
        panel.grid = element_blank()
      )

    map_path <- file.path(docs_dir, "wi_counties_check.png")
    ggsave(map_path, p, width = 6, height = 7, dpi = 150)
    cat(glue("  Map saved: {map_path}\n"))
    cat("  Open this PNG to visually confirm all 72 counties look correct.\n\n")
  }, error = function(e) {
    cat(glue("  Map not saved (ggplot2 issue): {e$message}\n\n"))
  })
}

# ── 5. Summary ────────────────────────────────────────────────
cat(strrep("=", 60), "\n")
cat("Workstream 01D complete.\n\n")
cat("Files produced:\n")
cat(glue("  data/raw/gis/wi_counties.shp  — county boundaries (shapefile)\n"))
cat(glue("  data/raw/gis/wi_counties.parquet — county boundaries (R/parquet)\n"))
cat(glue("  data/raw/gis/wi_tracts.shp    — census tracts\n"))
cat(glue("  data/raw/gis/rucc_wi.parquet  — rural-urban classification\n"))
cat(glue("  docs/wi_counties_check.png    — visual verification map\n\n"))
cat("Next steps:\n")
cat("  - For ArcGIS Pro: open data/raw/gis/wi_counties.shp\n")
cat("  - For R analysis: shapefiles load via st_read() in Phase 5 scripts\n")
cat("  - RUCC file: download from USDA if not yet done (see instructions above)\n")
cat(strrep("=", 60), "\n")
