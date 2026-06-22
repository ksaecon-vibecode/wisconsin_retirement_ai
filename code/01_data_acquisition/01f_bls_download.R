# ============================================================
# Workstream 1F: BLS Local Area Unemployment Statistics (LAUS)
# Annual County Unemployment Rates — Wisconsin 72 Counties
# ============================================================
# Project: Wisconsin Retirement AI
# Script:  code/01_data_acquisition/01f_bls_download.R
# Author:  Khawaja Sazzad Ali
# Date:    Summer 2026
#
# FIXES FROM v1:
#   - Rewrote BLS API JSON parser to handle actual response structure
#   - Added response inspection so we can see what the API returns
#   - Added manual Excel fallback with clear download instructions
#   - Removed httr dependency (uses jsonlite + base R instead)
#
# OUTPUT: data/processed/bls_unemployment_county_year.parquet
# ============================================================

library(tidyverse)
library(arrow)
library(here)
library(jsonlite)

# ── 0. Paths ─────────────────────────────────────────────────
proc_dir <- here("data", "processed")
raw_dir  <- here("data", "raw", "bls")
docs_dir <- here("docs")
dir.create(proc_dir,  showWarnings = FALSE, recursive = TRUE)
dir.create(raw_dir,   showWarnings = FALSE, recursive = TRUE)
dir.create(docs_dir,  showWarnings = FALSE, recursive = TRUE)

YEARS <- 2009:2024

# ── 1. Wisconsin county FIPS codes ───────────────────────────
WI_COUNTIES <- tribble(
  ~county_name,     ~county_fips,
  "Adams",          "55001", "Ashland",       "55003",
  "Barron",         "55005", "Bayfield",      "55007",
  "Brown",          "55009", "Buffalo",       "55011",
  "Burnett",        "55013", "Calumet",       "55015",
  "Chippewa",       "55017", "Clark",         "55019",
  "Columbia",       "55021", "Crawford",      "55023",
  "Dane",           "55025", "Dodge",         "55027",
  "Door",           "55029", "Douglas",       "55031",
  "Dunn",           "55033", "Eau Claire",    "55035",
  "Florence",       "55037", "Fond du Lac",   "55039",
  "Forest",         "55041", "Grant",         "55043",
  "Green",          "55045", "Green Lake",    "55047",
  "Iowa",           "55049", "Iron",          "55051",
  "Jackson",        "55053", "Jefferson",     "55055",
  "Juneau",         "55057", "Kenosha",       "55059",
  "Kewaunee",       "55061", "La Crosse",     "55063",
  "Lafayette",      "55065", "Langlade",      "55067",
  "Lincoln",        "55069", "Manitowoc",     "55071",
  "Marathon",       "55073", "Marinette",     "55075",
  "Marquette",      "55077", "Menominee",     "55078",
  "Milwaukee",      "55079", "Monroe",        "55081",
  "Oconto",         "55083", "Oneida",        "55085",
  "Outagamie",      "55087", "Ozaukee",       "55089",
  "Pepin",          "55091", "Pierce",        "55093",
  "Polk",           "55095", "Portage",       "55097",
  "Price",          "55099", "Racine",        "55101",
  "Richland",       "55103", "Rock",          "55105",
  "Rusk",           "55107", "St. Croix",     "55109",
  "Sauk",           "55111", "Sawyer",        "55113",
  "Shawano",        "55115", "Sheboygan",     "55117",
  "Taylor",         "55119", "Trempealeau",   "55121",
  "Vernon",         "55123", "Vilas",         "55125",
  "Walworth",       "55127", "Washburn",      "55129",
  "Washington",     "55131", "Waukesha",      "55133",
  "Waupaca",        "55135", "Waushara",      "55137",
  "Winnebago",      "55139", "Wood",          "55141"
)

# ── 2. Build BLS series IDs ───────────────────────────────────
# Format: LAUCN{SS}{CCC}0000000003
#   SS  = 2-digit state FIPS (55 for Wisconsin)
#   CCC = 3-digit county FIPS
#   003 = unemployment rate series type
WI_COUNTIES <- WI_COUNTIES %>%
  mutate(
    county_3  = str_sub(county_fips, 4, 6),
    series_id = paste0("LAUCN55", county_3, "0000000003")
  )

cat("Sample series IDs:\n")
print(head(WI_COUNTIES %>% select(county_name, series_id), 3))

# ── 3. BLS API download function (fixed parser) ───────────────
download_bls_batch <- function(series_ids, start_year, end_year) {
  
  payload <- toJSON(list(
    seriesid      = as.list(series_ids),
    startyear     = as.character(start_year),
    endyear       = as.character(end_year),
    annualaverage = TRUE,
    calculations  = FALSE
  ), auto_unbox = TRUE)
  
  # Use base R's url() + readLines to avoid httr dependency issues
  resp_text <- tryCatch({
    con <- url("https://api.bls.gov/publicAPI/v2/timeseries/data/",
               open = "r")
    on.exit(close(con), add = TRUE)
    NULL  # Can't POST with base url() — use alternative below
  }, error = function(e) NULL)
  
  # Use jsonlite + curl approach
  resp_text <- tryCatch({
    result <- system2(
      "curl",
      args = c(
        "-s",
        "-X", "POST",
        "-H", "'Content-Type: application/json'",
        "-d", shQuote(payload),
        "https://api.bls.gov/publicAPI/v2/timeseries/data/"
      ),
      stdout = TRUE,
      stderr = FALSE
    )
    paste(result, collapse = "")
  }, error = function(e) NULL)
  
  # If curl not available, try httr
  if (is.null(resp_text) || nchar(resp_text) < 10) {
    resp_text <- tryCatch({
      if (requireNamespace("httr", quietly = TRUE)) {
        r <- httr::POST(
          "https://api.bls.gov/publicAPI/v2/timeseries/data/",
          httr::add_headers("Content-Type" = "application/json"),
          body = payload,
          encode = "raw",
          httr::timeout(60)
        )
        httr::content(r, "text", encoding = "UTF-8")
      } else NULL
    }, error = function(e) NULL)
  }
  
  if (is.null(resp_text) || nchar(resp_text) < 10) {
    cat("    Cannot reach BLS API\n")
    return(NULL)
  }
  
  # ── Parse JSON response ───────────────────────────────────
  parsed <- tryCatch(fromJSON(resp_text, simplifyVector = FALSE),
                     error = function(e) NULL)
  if (is.null(parsed)) {
    cat("    JSON parse failed\n")
    return(NULL)
  }
  
  # Inspect status
  status <- parsed$status
  cat(sprintf("    API status: %s\n", status))
  
  if (!identical(status, "REQUEST_SUCCEEDED")) {
    msgs <- parsed$message
    cat(sprintf("    Message: %s\n", paste(unlist(msgs), collapse = "; ")))
    return(NULL)
  }
  
  # ── Extract series data ───────────────────────────────────
  # BLS response structure (with simplifyVector=FALSE):
  # parsed$Results$series is a LIST of series objects
  # Each series: list(seriesID="...", data=list(...))
  # Each data item: list(year="2021", period="M13", value="4.5", ...)
  # period "M13" = annual average
  
  series_list <- parsed$Results$series
  if (is.null(series_list) || length(series_list) == 0) {
    cat("    No series data in response\n")
    return(NULL)
  }
  
  cat(sprintf("    Series returned: %d\n", length(series_list)))
  
  # Extract records from each series
  records <- list()
  for (s in series_list) {
    sid   <- s$seriesID
    sdata <- s$data
    
    if (is.null(sdata) || length(sdata) == 0) next
    
    for (d in sdata) {
      # Annual average = period M13
      if (!identical(d$period, "M13")) next
      val <- suppressWarnings(as.numeric(d$value))
      if (is.na(val)) next
      
      records[[length(records) + 1]] <- data.frame(
        series_id  = as.character(sid),
        year       = as.integer(d$year),
        unemp_rate = val,
        stringsAsFactors = FALSE
      )
    }
  }
  
  if (length(records) == 0) {
    cat("    No annual average records found (M13 period)\n")
    return(NULL)
  }
  
  bind_rows(records)
}

# ── 4. Run API download ───────────────────────────────────────
cat(strrep("=", 60), "\n")
cat("Workstream 1F: BLS LAUS Unemployment\n")
cat(strrep("=", 60), "\n\n")

all_series  <- WI_COUNTIES$series_id
BATCH_SIZE  <- 25    # Conservative batch size for reliability
year_ranges <- list(c(2009, 2016), c(2017, 2024))

all_results  <- list()
api_worked   <- FALSE

for (yr in year_ranges) {
  for (i in seq(1, length(all_series), by = BATCH_SIZE)) {
    batch <- all_series[i:min(i + BATCH_SIZE - 1, length(all_series))]
    cat(sprintf("  Downloading %d series, years %d-%d...\n",
                length(batch), yr[1], yr[2]))
    
    result <- download_bls_batch(batch, yr[1], yr[2])
    
    if (!is.null(result) && nrow(result) > 0) {
      all_results <- c(all_results, list(result))
      cat(sprintf("    Got %d records\n", nrow(result)))
      api_worked <- TRUE
    } else {
      cat("    No data returned for this batch\n")
    }
    Sys.sleep(2)  # Be polite to BLS API
  }
}

# ── 5. Combine API results or fall back to manual ─────────────
if (api_worked && length(all_results) > 0) {
  
  df_bls <- bind_rows(all_results) %>%
    left_join(
      WI_COUNTIES %>% select(county_name, county_fips, series_id),
      by = "series_id"
    ) %>%
    filter(!is.na(county_fips)) %>%
    select(county_fips, county_name, year, unemp_rate) %>%
    arrange(county_fips, year)
  
  cat(sprintf("\nAPI download complete: %d county-year records\n", nrow(df_bls)))
  
} else {
  
  # ── Manual download fallback ─────────────────────────────
  cat("\n")
  cat(strrep("-", 60), "\n")
  cat("BLS API unavailable. Using manual download fallback.\n")
  cat(strrep("-", 60), "\n")
  cat("\nDownload BLS county unemployment files manually:\n")
  cat("  1. Go to: https://www.bls.gov/lau/tables.htm\n")
  cat("  2. Click 'County and metropolitan area annual averages'\n")
  cat("  3. For each year 2009-2024, click the link and download\n")
  cat("  4. Save each as: data/raw/bls/laucnty_{YEAR}.xlsx\n")
  cat("     Example: data/raw/bls/laucnty_2021.xlsx\n\n")
  
  # Check for manually downloaded files
  # BLS names files as laucnty09.xlsx through laucnty25.xlsx (2-digit year)
  # Also accepts laucnty_2009.xlsx format as fallback
  bls_files <- list.files(
    raw_dir,
    pattern = "laucnty\\d{2,4}\\.(xlsx|xls|csv)$",
    full.names = TRUE,
    ignore.case = TRUE
  )
  
  if (length(bls_files) == 0) {
    cat("No manual BLS files found in data/raw/bls/\n")
    cat("Downloading manually and saving to that folder.\n")
    cat("Then re-run this script.\n\n")
    
    # Create a placeholder so other scripts can proceed
    df_placeholder <- WI_COUNTIES %>%
      select(county_fips, county_name) %>%
      cross_join(tibble(year = YEARS)) %>%
      mutate(unemp_rate = NA_real_)
    
    out_path <- file.path(proc_dir, "bls_unemployment_county_year.parquet")
    write_parquet(df_placeholder, out_path)
    cat(sprintf("Placeholder saved: %s\n", out_path))
    cat("Re-run after downloading BLS files.\n")
    stop("BLS data not yet available. See instructions above.")
  }
  
  cat(sprintf("Found %d manual BLS files. Processing...\n",
              length(bls_files)))
  
  df_bls <- map_dfr(bls_files, function(f) {
    # Extract year from filename — handles both formats:
    #   laucnty09.xlsx  -> 2-digit -> add 2000 (09 -> 2009)
    #   laucnty25.xlsx  -> 2-digit -> add 2000 (25 -> 2025)
    #   laucnty_2021.xlsx -> 4-digit -> use as-is
    yr_raw <- str_extract(basename(f), "\\d{2,4}")
    yr <- if (nchar(yr_raw) == 2) {
      as.integer(yr_raw) + 2000L
    } else {
      as.integer(yr_raw)
    }
    cat(sprintf("  Reading %s (year=%d)...\n", basename(f), yr))
    
    ext <- tolower(tools::file_ext(f))
    
    # ── Read with NO header (col_names=FALSE) ─────────────────
    # BLS files have NO header row. First data row looks like:
    # CN0100500000000 | 01 | 005 | Barbour County, AL | 2009 | 10000 | 8728 | 1272 | 12.7
    # Columns (fixed positions):
    #   1 = Series code (CN...)
    #   2 = State FIPS (2-digit)
    #   3 = County FIPS (3-digit)
    #   4 = County Name, State abbreviation
    #   5 = Year
    #   6 = Labor force
    #   7 = Employed
    #   8 = Unemployed
    #   9 = Unemployment rate (%)
    
    tryCatch({
      if (ext == "csv") {
        df_raw <- read_csv(
          f,
          col_names = TRUE,
          show_col_types = FALSE,
          skip = 2        # Row 1 = title, Row 2 = headers, data from Row 3
        )
      } else {
        if (!requireNamespace("readxl", quietly = TRUE)) {
          install.packages("readxl")
        }
        df_raw <- readxl::read_excel(
          f,
          skip = 1,       # Skip title row; Row 2 becomes header
          col_names = TRUE
        )
      }
      
      # Standardize column names to lowercase with underscores
      names(df_raw) <- names(df_raw) %>%
        str_to_lower() %>%
        str_replace_all("[^a-z0-9]", "_") %>%
        str_replace_all("_+", "_") %>%
        str_remove("^_") %>%
        str_remove("_$")
      
      cat(sprintf("    Columns: %s\n",
                  paste(names(df_raw), collapse = ", ")))
      
      # Confirmed column names after cleaning:
      #   laus_code, state_fips_code, county_fips_code,
      #   county_name_state_abbreviation, year,
      #   labor_force, employed, unemployed, unemployment_rate
      # We identify by position as fallback if name matching fails
      
      # Find state FIPS column (col 2)
      state_col <- names(df_raw)[str_detect(names(df_raw),
                                            "state.*fips|statefips")]
      if (length(state_col) == 0) state_col <- names(df_raw)[2]
      
      # Find county FIPS column (col 3)
      county_col <- names(df_raw)[str_detect(names(df_raw),
                                             "county.*fips|countyfips")]
      if (length(county_col) == 0) county_col <- names(df_raw)[3]
      
      # Find unemployment rate column (col 9)
      rate_col <- names(df_raw)[str_detect(names(df_raw),
                                           "unemployment_rate|rate")]
      if (length(rate_col) == 0) rate_col <- names(df_raw)[9]
      
      cat(sprintf("    State col: '%s' | County col: '%s' | Rate col: '%s'\n",
                  state_col[1], county_col[1], rate_col[1]))
      
      # Build 5-digit county FIPS and filter to Wisconsin (55)
      df_raw <- df_raw %>%
        mutate(
          state_fips_std  = str_pad(as.character(.data[[state_col[1]]]),
                                    2, "left", "0"),
          county_fips_std = str_pad(as.character(.data[[county_col[1]]]),
                                    3, "left", "0"),
          county_fips     = paste0(state_fips_std, county_fips_std),
          year            = yr,
          unemp_rate      = suppressWarnings(
            as.numeric(.data[[rate_col[1]]]))
        )
      
      df_wi <- df_raw %>%
        filter(state_fips_std == "55") %>%
        select(county_fips, year, unemp_rate) %>%
        filter(!is.na(unemp_rate))
      
      cat(sprintf("    Wisconsin rows: %d\n", nrow(df_wi)))
      return(df_wi)
      
    }, error = function(e) {
      cat(sprintf("    ERROR: %s\n", e$message))
      return(NULL)
    })
  })
  
  df_bls <- df_bls %>%
    left_join(
      WI_COUNTIES %>% select(county_name, county_fips),
      by = "county_fips"
    ) %>%
    filter(!is.na(county_name)) %>%
    select(county_fips, county_name, year, unemp_rate) %>%
    arrange(county_fips, year)
  
  cat(sprintf("Manual files produced %d records\n", nrow(df_bls)))
}

# ── 6. Quality checks ─────────────────────────────────────────
cat("\nUnemployment rate summary (Wisconsin counties):\n")
print(summary(df_bls$unemp_rate))

expected <- nrow(WI_COUNTIES) * length(YEARS)
actual   <- nrow(df_bls)
cat(sprintf("\nExpected county-year cells: %d\n", expected))
cat(sprintf("Actual county-year cells:   %d\n", actual))
cat(sprintf("Coverage:                   %.1f%%\n",
            actual / expected * 100))

# Check a known value: Wisconsin statewide unemployment ~3.5% in 2019
wi_2019 <- df_bls %>% filter(year == 2019)
if (nrow(wi_2019) > 0) {
  cat(sprintf("\nSanity check — Mean WI county unemp rate 2019: %.2f%%\n",
              mean(wi_2019$unemp_rate, na.rm = TRUE)))
  cat("(Expected: ~3-4% for pre-COVID Wisconsin)\n")
}

# ── 7. Save ───────────────────────────────────────────────────
out_path <- file.path(proc_dir, "bls_unemployment_county_year.parquet")
write_parquet(df_bls, out_path)
cat(sprintf("\nSaved: %s\n", out_path))

cat("\n", strrep("=", 60), "\n", sep="")
cat("Workstream 1F complete.\n")
cat(strrep("=", 60), "\n", sep="")