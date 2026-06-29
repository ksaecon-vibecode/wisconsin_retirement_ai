# ============================================================
# Phase 5 — Script 05b: County-Level Maps
# Four analytical maps for the paper
# ============================================================
# Project: Wisconsin Retirement AI
# Script:  code/05_gis/05b_county_maps.R
# Author:  Khawaja Sazzad Ali
# Date:    Summer 2026
#
# WHAT THIS SCRIPT DOES:
#   Produces four choropleth maps showing the spatial distribution
#   of the study's key variables across Wisconsin's 72 counties.
#   These are the analytical figures that go directly into the paper.
#
#   A choropleth map colors each county according to a data value.
#   Darker colors = higher values (or worse outcomes, depending on
#   the variable). The color scale is chosen to highlight variation.
#
# THE FOUR MAPS:
#   Map 1 — Retirement enrollment rate by county
#            (the outcome variable, shows where the problem is worst)
#   Map 2 — Financial literacy score by county
#            (one behavioral barrier)
#   Map 3 — AI complaint density by county
#            (the AI exposure variable)
#   Map 4 — Broadband providers by county
#            (the instrumental variable)
#
# RUN THIS AFTER: Phase 4 econometric models
#   (county-level summary produced by 05a_spatial_diagnostics.R)
#
# OUTPUT:
#   outputs/figures/map_enrollment_rate.png
#   outputs/figures/map_literacy_score.png
#   outputs/figures/map_ai_density.png
#   outputs/figures/map_broadband_iv.png
#   outputs/figures/maps_four_panel.png  (combined figure for paper)
# ============================================================

library(tidyverse)
library(sf)
library(arrow)
library(here)
library(glue)
library(viridis)      # Color scales for maps
library(patchwork)    # Combine multiple plots

# ── 0. Setup ─────────────────────────────────────────────────
gis_dir    <- here("data", "raw", "gis")
proc_dir   <- here("data", "processed")
final_dir  <- here("data", "final")
fig_dir    <- here("outputs", "figures")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

cat(strrep("=", 60), "\n")
cat("Phase 5 — County Maps\n")
cat(format(Sys.time()), "\n")
cat(strrep("=", 60), "\n\n")

# ── 1. Load shapefile ─────────────────────────────────────────
cat("Loading Wisconsin county shapefile...\n")
wi_sf <- st_read(file.path(gis_dir, "wi_counties.shp"), quiet = TRUE) %>%
  mutate(county_fips = GEOID)

# Project to a WI-specific coordinate system for better map appearance
# EPSG 3071 = NAD83 / Wisconsin Transverse Mercator (official WI system)
wi_sf <- st_transform(wi_sf, crs = 3071)
cat(glue("  {nrow(wi_sf)} counties loaded, projected to WI Transverse Mercator\n\n"))

# ── 2. Load data ──────────────────────────────────────────────
cat("Loading county-level summary data...\n")

county_summary_path <- file.path(proc_dir, "county_level_summary.parquet")
if (!file.exists(county_summary_path)) {
  cat("  county_level_summary.parquet not found.\n")
  cat("  Computing from master dataset...\n")

  master <- read_parquet(file.path(final_dir, "master_dataset.parquet"))
  county_summary <- master %>%
    group_by(county_fips) %>%
    summarise(
      enroll_rate     = mean(enrolled, na.rm = TRUE) * 100,
      mean_literacy   = mean(literacy_score, na.rm = TRUE),
      mean_impatience = mean(impatience_index, na.rm = TRUE),
      mean_ai_density = mean(ai_complaint_density_final, na.rm = TRUE),
      mean_distress   = mean(distress_index, na.rm = TRUE),
      mean_bb         = mean(bb_providers_25_3, na.rm = TRUE),
      mean_poverty    = mean(poverty_rate, na.rm = TRUE),
      n_respondents   = n(),
      .groups = "drop"
    )
} else {
  county_summary <- read_parquet(county_summary_path) %>%
    mutate(enroll_rate = enroll_rate * 100)
}

# Load RUCC rural-urban classification
rucc_path <- file.path(gis_dir, "rucc_wi.parquet")
if (file.exists(rucc_path)) {
  rucc_wi <- read_parquet(rucc_path)
  county_summary <- county_summary %>%
    left_join(rucc_wi %>% select(county_fips, rural_class), by = "county_fips")
} else {
  county_summary$rural_class <- NA_character_
}

cat(glue("  {nrow(county_summary)} counties with data\n\n"))

# ── 3. Join data to shapefile ─────────────────────────────────
wi_map <- wi_sf %>%
  left_join(county_summary, by = "county_fips")

# ── 4. Map theme ──────────────────────────────────────────────
# Consistent styling across all four maps
theme_wi_map <- function() {
  theme_void() +
  theme(
    plot.title       = element_text(size = 12, face = "bold", hjust = 0.5,
                                    margin = margin(b = 4)),
    plot.subtitle    = element_text(size = 9, hjust = 0.5, color = "#555555",
                                    margin = margin(b = 8)),
    plot.caption     = element_text(size = 7, color = "#888888", hjust = 1,
                                    margin = margin(t = 6)),
    legend.position  = "bottom",
    legend.title     = element_text(size = 8, face = "bold"),
    legend.text      = element_text(size = 7),
    legend.key.width = unit(1.5, "cm"),
    legend.key.height= unit(0.3, "cm"),
    plot.background  = element_rect(fill = "white", color = NA),
    plot.margin      = margin(8, 8, 8, 8)
  )
}

# County border styling
COUNTY_BORDER_COLOR <- "#999999"
COUNTY_BORDER_SIZE  <- 0.2
NO_DATA_COLOR       <- "#e0e0e0"   # Gray for counties with no data

# ── 5. Map 1 — Retirement Enrollment Rate ────────────────────
cat("Producing Map 1: Retirement enrollment rate...\n")

map1 <- ggplot(wi_map) +
  geom_sf(
    aes(fill = enroll_rate),
    color    = COUNTY_BORDER_COLOR,
    linewidth= COUNTY_BORDER_SIZE
  ) +
  scale_fill_viridis_c(
    name    = "Enrollment rate (%)",
    option  = "D",       # Blue-yellow-green palette
    na.value= NO_DATA_COLOR,
    labels  = function(x) paste0(round(x), "%")
  ) +
  labs(
    title    = "Retirement Plan Enrollment Rate",
    subtitle = "Wisconsin counties — NFCS respondents (pooled 2009–2024)",
    caption  = "Source: FINRA NFCS restricted-use data | Gray = no respondents in county"
  ) +
  theme_wi_map()

ggsave(file.path(fig_dir, "map_enrollment_rate.png"), map1,
       width = 7, height = 8, dpi = 300)
cat("  Saved: map_enrollment_rate.png\n")

# ── 6. Map 2 — Financial Literacy Score ──────────────────────
cat("Producing Map 2: Financial literacy score...\n")

map2 <- ggplot(wi_map) +
  geom_sf(
    aes(fill = mean_literacy),
    color    = COUNTY_BORDER_COLOR,
    linewidth= COUNTY_BORDER_SIZE
  ) +
  scale_fill_viridis_c(
    name     = "Mean literacy score (0–5)",
    option   = "C",      # Yellow-orange-purple palette
    na.value = NO_DATA_COLOR,
    limits   = c(2, 4),
    labels   = function(x) sprintf("%.1f", x)
  ) +
  labs(
    title    = "Financial Literacy Score",
    subtitle = "Mean Big Five score by county — lower = more vulnerable",
    caption  = "Source: FINRA NFCS | Gray = no respondents in county"
  ) +
  theme_wi_map()

ggsave(file.path(fig_dir, "map_literacy_score.png"), map2,
       width = 7, height = 8, dpi = 300)
cat("  Saved: map_literacy_score.png\n")

# ── 7. Map 3 — AI Complaint Density ──────────────────────────
cat("Producing Map 3: AI complaint density...\n")

# For counties with zero AI density (most rural counties),
# use 0 explicitly so they appear in the scale, not as missing
wi_map_ai <- wi_map %>%
  mutate(
    ai_density_plot = if_else(
      is.na(mean_ai_density), 0, mean_ai_density
    ),
    has_cfpb_data = !is.na(mean_ai_density)
  )

map3 <- ggplot(wi_map_ai) +
  geom_sf(
    aes(fill = ai_density_plot),
    color    = COUNTY_BORDER_COLOR,
    linewidth= COUNTY_BORDER_SIZE
  ) +
  scale_fill_viridis_c(
    name     = "AI complaints per 10,000 residents",
    option   = "B",      # Magenta-yellow palette
    na.value = NO_DATA_COLOR,
    trans    = "sqrt",   # Square root transform to show low-value variation
    labels   = function(x) sprintf("%.3f", x)
  ) +
  labs(
    title    = "AI Financial Tool Complaint Density",
    subtitle = "CFPB complaints mentioning AI financial tools (2015–2025)\nCounties with no CFPB data assigned zero density",
    caption  = "Source: CFPB Consumer Complaint Database — NLP semantic classifier"
  ) +
  theme_wi_map()

ggsave(file.path(fig_dir, "map_ai_density.png"), map3,
       width = 7, height = 8, dpi = 300)
cat("  Saved: map_ai_density.png\n")

# ── 8. Map 4 — Broadband IV ──────────────────────────────────
cat("Producing Map 4: Broadband penetration (IV)...\n")

map4 <- ggplot(wi_map) +
  geom_sf(
    aes(fill = mean_bb),
    color    = COUNTY_BORDER_COLOR,
    linewidth= COUNTY_BORDER_SIZE
  ) +
  scale_fill_viridis_c(
    name     = "ISPs offering 25/3 Mbps",
    option   = "E",      # Turquoise palette
    na.value = NO_DATA_COLOR,
    labels   = function(x) sprintf("%.1f", x)
  ) +
  labs(
    title    = "Broadband Penetration (IV for AI Exposure)",
    subtitle = "Number of ISPs offering 25/3 Mbps residential service\nFCC Form 477 — instrumental variable for AI exposure endogeneity",
    caption  = "Source: FCC Form 477 County-Level Tier Data"
  ) +
  theme_wi_map()

ggsave(file.path(fig_dir, "map_broadband_iv.png"), map4,
       width = 7, height = 8, dpi = 300)
cat("  Saved: map_broadband_iv.png\n\n")

# ── 9. Four-panel combined figure ────────────────────────────
cat("Producing four-panel combined figure for paper...\n")

# Add panel labels
map1_p <- map1 + labs(tag = "A")
map2_p <- map2 + labs(tag = "B")
map3_p <- map3 + labs(tag = "C")
map4_p <- map4 + labs(tag = "D")

combined <- (map1_p | map2_p) / (map3_p | map4_p) +
  plot_annotation(
    title   = "Figure 1: Wisconsin County-Level Distribution of Key Variables",
    caption = paste0(
      "Note: Panel A shows retirement plan enrollment rates. ",
      "Panel B shows mean financial literacy (Big Five score). ",
      "Panel C shows AI financial tool complaint density per 10,000 residents. ",
      "Panel D shows broadband penetration used as the instrumental variable. ",
      "Gray counties have no NFCS respondents (A-B) or no CFPB complaint data (C)."
    ),
    theme = theme(
      plot.title   = element_text(size = 13, face = "bold", hjust = 0.5),
      plot.caption = element_text(size = 7, color = "#666666", hjust = 0)
    )
  )

ggsave(file.path(fig_dir, "maps_four_panel.png"), combined,
       width = 14, height = 16, dpi = 300)
cat("  Saved: maps_four_panel.png\n")

# ── 10. Summary ───────────────────────────────────────────────
cat("\n", strrep("=", 60), "\n", sep="")
cat("Maps complete. Files saved to outputs/figures/\n")
cat("  map_enrollment_rate.png\n")
cat("  map_literacy_score.png\n")
cat("  map_ai_density.png\n")
cat("  map_broadband_iv.png\n")
cat("  maps_four_panel.png  <- USE THIS IN THE PAPER\n\n")
cat("For ArcGIS Pro versions:\n")
cat("  Import county_level_summary.parquet data\n")
cat("  Join to data/raw/gis/wi_counties.shp via county_fips\n")
cat("  Apply professional cartographic styling\n")
cat(strrep("=", 60), "\n")
