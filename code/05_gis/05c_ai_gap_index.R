# ============================================================
# Phase 5 — Script 05c: Wisconsin AI Gap Index
# County-Level Classification and Policy Map
# ============================================================
# Project: Wisconsin Retirement AI
# Script:  code/05_gis/05c_ai_gap_index.R
# Author:  Khawaja Sazzad Ali
# Date:    Summer 2026
#
# WHAT THIS SCRIPT DOES:
#   Constructs the Wisconsin AI Gap Index — the primary policy
#   deliverable of the project. Scores all 72 Wisconsin counties
#   on two dimensions and classifies them into four quadrants:
#
#   DIMENSION 1 — Behavioral Barrier Severity (x-axis)
#     How difficult is it for workers in this county to enroll
#     in retirement savings plans due to behavioral barriers?
#     Measured by: low literacy rate + high impatience rate,
#     weighted by the coefficients from the Probit models.
#     Higher score = more severe behavioral barriers.
#
#   DIMENSION 2 — AI Financial Tool Exposure (y-axis)
#     How much AI financial tool presence exists in this county?
#     Measured by: AI complaint density from CFPB NLP pipeline.
#     Higher score = more AI tool exposure.
#     Counties with no CFPB data = zero exposure by assumption.
#
#   FOUR QUADRANTS:
#   Q1: High Barrier + Low AI  = "Underserved" — highest priority
#       These counties have the worst behavioral barriers AND the
#       least AI tool presence. They need intervention most urgently.
#
#   Q2: High Barrier + High AI = "Emerging" — monitor carefully
#       These counties have bad barriers but some AI presence.
#       The AI tools may be starting to help but need support.
#
#   Q3: Low Barrier + Low AI   = "Self-sufficient" — low priority
#       These counties have mild behavioral barriers and don't
#       need AI intervention as urgently.
#
#   Q4: Low Barrier + High AI  = "Leaders" — study and replicate
#       These counties have the best outcomes and the most AI
#       tool presence. Learn what works here and replicate it.
#
# RUN THIS AFTER:
#   Phase 4 econometric models (for barrier severity weights)
#   Phase 5 scripts 05a and 05b
#
# OUTPUT:
#   data/processed/wi_ai_gap_index.parquet  <- county scores
#   outputs/figures/map_ai_gap_index.png    <- main policy map
#   outputs/tables/ai_gap_index_rankings.csv
#   outputs/figures/ai_gap_index_scatter.png <- quadrant plot
# ============================================================

library(tidyverse)
library(sf)
library(arrow)
library(here)
library(glue)
library(viridis)
library(scales)    # For axis formatting

# ── 0. Setup ─────────────────────────────────────────────────
gis_dir   <- here("data", "raw", "gis")
proc_dir  <- here("data", "processed")
final_dir <- here("data", "final")
fig_dir   <- here("outputs", "figures")
tbl_dir   <- here("outputs", "tables")
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(tbl_dir, showWarnings = FALSE, recursive = TRUE)

cat(strrep("=", 60), "\n")
cat("Phase 5 — Wisconsin AI Gap Index\n")
cat(format(Sys.time()), "\n")
cat(strrep("=", 60), "\n\n")

# ── 1. Load data ──────────────────────────────────────────────
cat("Loading county data...\n")

# Master dataset (for county-level aggregation if needed)
master <- read_parquet(file.path(final_dir, "master_dataset.parquet"))

# County shapefile
wi_sf <- st_read(file.path(gis_dir, "wi_counties.shp"), quiet = TRUE) %>%
  mutate(county_fips = GEOID) %>%
  st_transform(crs = 3071)

# RUCC rural-urban classification
rucc_path <- file.path(gis_dir, "rucc_wi.parquet")
rucc_wi <- if (file.exists(rucc_path)) read_parquet(rucc_path) else NULL

cat(glue("  Master: {format(nrow(master), big.mark=',')} respondents\n"))
cat(glue("  Counties in shapefile: {nrow(wi_sf)}\n\n"))

# ── 2. Compute county-level index components ──────────────────
cat("Computing index components...\n")

# Aggregate from individual-level master dataset to county level
# Using the most recent available year per county for current state
county_data <- master %>%
  # Use all waves for stability (more observations = more reliable estimates)
  group_by(county_fips) %>%
  summarise(
    county_name      = first(county_fips),  # Will join name from shapefile
    n_respondents    = n(),

    # DIMENSION 1 components (behavioral barriers)
    enroll_rate      = mean(enrolled, na.rm = TRUE),
    mean_literacy    = mean(literacy_score, na.rm = TRUE),
    mean_impatience  = mean(impatience_index, na.rm = TRUE),
    pct_low_literacy = mean(literacy_score <= 2, na.rm = TRUE),
    pct_high_impatience = mean(impatience_index >= 2, na.rm = TRUE),

    # DIMENSION 2 components (AI exposure)
    # Use most recent year's AI density (2024 or 2025)
    ai_density       = mean(ai_complaint_density_final, na.rm = TRUE),
    ai_density_log   = mean(ai_complaint_density_final_log, na.rm = TRUE),
    mean_bb          = mean(bb_providers_25_3, na.rm = TRUE),

    # Controls for context
    mean_poverty     = mean(poverty_rate, na.rm = TRUE),
    mean_income      = mean(log_median_income, na.rm = TRUE),
    pct_minority     = mean(minority, na.rm = TRUE),

    .groups = "drop"
  ) %>%
  filter(!is.na(county_fips))

cat(glue("  Counties with respondent data: {nrow(county_data)}\n"))

# For counties WITHOUT respondents, use state averages for barrier severity
# (they still appear on the map, just with imputed values)
all_counties <- wi_sf %>%
  as.data.frame() %>%
  select(county_fips, NAME) %>%
  rename(county_name_official = NAME)

county_data <- all_counties %>%
  left_join(county_data, by = "county_fips") %>%
  mutate(
    has_respondent_data = !is.na(n_respondents),
    # For counties without respondents, use state medians
    enroll_rate      = if_else(is.na(enroll_rate),
                               median(county_data$enroll_rate, na.rm=TRUE),
                               enroll_rate),
    mean_literacy    = if_else(is.na(mean_literacy),
                               median(county_data$mean_literacy, na.rm=TRUE),
                               mean_literacy),
    mean_impatience  = if_else(is.na(mean_impatience),
                               median(county_data$mean_impatience, na.rm=TRUE),
                               mean_impatience),
    # Counties without CFPB data = zero AI density
    ai_density       = if_else(is.na(ai_density), 0, ai_density),
    ai_density_log   = if_else(is.na(ai_density_log), 0, ai_density_log)
  )

# ── 3. Construct the two index dimensions ─────────────────────
cat("\nConstructing index dimensions...\n")

# DIMENSION 1 — Behavioral Barrier Severity
# Combines low enrollment (outcome), low literacy (barrier), and
# high impatience (barrier) into a composite barrier score.
# Each component is standardized (z-score) then averaged.
#
# Note: enrollment rate is INVERTED so that higher score = worse barriers
# (low enrollment = high barrier severity)

county_data <- county_data %>%
  mutate(
    # Standardize each component (subtract mean, divide by SD)
    z_enroll_inv  = -scale(enroll_rate)[,1],   # Inverted: low enroll = high barrier
    z_literacy_inv= -scale(mean_literacy)[,1], # Inverted: low literacy = high barrier
    z_impatience  =  scale(mean_impatience)[,1], # High impatience = high barrier

    # Composite barrier severity (average of three standardized components)
    barrier_severity = (z_enroll_inv + z_literacy_inv + z_impatience) / 3,

    # DIMENSION 2 — AI Financial Tool Exposure
    # Uses log AI density (main measure) + broadband penetration (supports AI access)
    z_ai_density  = scale(ai_density_log)[,1],
    z_broadband   = scale(mean_bb)[,1],

    # Composite AI exposure (weighted: 70% AI density, 30% broadband)
    # Broadband is the enabling infrastructure; AI density is the outcome
    ai_exposure   = 0.7 * z_ai_density + 0.3 * z_broadband
  )

# ── 4. Assign quadrant classifications ────────────────────────
cat("Assigning quadrant classifications...\n")

# Split on zero (median of standardized scores)
county_data <- county_data %>%
  mutate(
    quadrant = case_when(
      barrier_severity >  0 & ai_exposure <= 0 ~ "Q1: Underserved",
      barrier_severity >  0 & ai_exposure >  0 ~ "Q2: Emerging",
      barrier_severity <= 0 & ai_exposure <= 0 ~ "Q3: Self-Sufficient",
      barrier_severity <= 0 & ai_exposure >  0 ~ "Q4: Leaders",
      TRUE ~ "Unclassified"
    ),
    # Policy priority order (Q1 = highest need)
    priority = case_when(
      quadrant == "Q1: Underserved"     ~ 1L,
      quadrant == "Q2: Emerging"        ~ 2L,
      quadrant == "Q3: Self-Sufficient" ~ 3L,
      quadrant == "Q4: Leaders"         ~ 4L,
      TRUE ~ 5L
    )
  )

# Quadrant counts
cat("\nQuadrant distribution:\n")
county_data %>%
  count(quadrant) %>%
  arrange(quadrant) %>%
  walk(~ cat(glue("  {.x$quadrant}: {.x$n} counties\n")))

# ── 5. Join RUCC classification ───────────────────────────────
if (!is.null(rucc_wi)) {
  county_data <- county_data %>%
    left_join(
      rucc_wi %>% select(county_fips, rucc_code, rural_class),
      by = "county_fips"
    )
}

# ── 6. Save index scores ──────────────────────────────────────
index_path <- file.path(proc_dir, "wi_ai_gap_index.parquet")
write_parquet(county_data, index_path)
cat(glue("\nIndex saved: {index_path}\n"))

# ── 7. Save rankings table (for paper and policy brief) ───────
rankings <- county_data %>%
  select(county_fips, county_name_official,
         barrier_severity, ai_exposure, quadrant, priority,
         enroll_rate, mean_literacy, mean_impatience,
         ai_density, mean_bb, mean_poverty,
         has_respondent_data) %>%
  arrange(priority, desc(barrier_severity)) %>%
  mutate(
    across(c(barrier_severity, ai_exposure, enroll_rate,
             mean_literacy, mean_impatience, ai_density,
             mean_bb, mean_poverty),
           ~ round(.x, 4))
  )

rankings_path <- file.path(tbl_dir, "ai_gap_index_rankings.csv")
write_csv(rankings, rankings_path)
cat(glue("Rankings saved: {rankings_path}\n\n"))

# ── 8. Quadrant scatter plot ───────────────────────────────────
cat("Producing quadrant scatter plot...\n")

# Color palette for quadrants
QUADRANT_COLORS <- c(
  "Q1: Underserved"      = "#d73027",   # Red — highest priority
  "Q2: Emerging"         = "#f46d43",   # Orange — watch carefully
  "Q3: Self-Sufficient"  = "#74add1",   # Blue — low priority
  "Q4: Leaders"          = "#313695",   # Dark blue — study and replicate
  "Unclassified"         = "#aaaaaa"
)

scatter <- ggplot(county_data,
       aes(x = barrier_severity, y = ai_exposure, color = quadrant)) +
  # Add quadrant shading
  annotate("rect", xmin=-Inf, xmax=0, ymin=0,    ymax=Inf,
           fill="#313695", alpha=0.05) +  # Q4 Leaders
  annotate("rect", xmin=0,   xmax=Inf, ymin=0,    ymax=Inf,
           fill="#f46d43", alpha=0.05) +  # Q2 Emerging
  annotate("rect", xmin=-Inf, xmax=0, ymin=-Inf, ymax=0,
           fill="#74add1", alpha=0.05) +  # Q3 Self-Sufficient
  annotate("rect", xmin=0,   xmax=Inf, ymin=-Inf, ymax=0,
           fill="#d73027", alpha=0.05) +  # Q1 Underserved
  # Reference lines at zero
  geom_hline(yintercept = 0, linetype = "dashed", color = "#666666", linewidth=0.5) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "#666666", linewidth=0.5) +
  # County points
  geom_point(size = 2.5, alpha = 0.85) +
  # Label counties with highest barrier severity (top priority)
  ggrepel::geom_text_repel(
    data = county_data %>% filter(barrier_severity > 0.5 | ai_exposure > 0.8),
    aes(label = county_name_official),
    size = 2.5, max.overlaps = 15,
    segment.color = "#888888", segment.size = 0.3
  ) +
  scale_color_manual(values = QUADRANT_COLORS, name = "Quadrant") +
  labs(
    title    = "Wisconsin AI Gap Index — County Quadrant Classification",
    subtitle = "72 counties scored on behavioral barrier severity and AI financial tool exposure",
    x        = "Behavioral Barrier Severity\n(higher = worse outcomes, lower literacy, more impatience)",
    y        = "AI Financial Tool Exposure\n(higher = more AI complaint density and broadband access)",
    caption  = paste0(
      "Note: Axes are standardized scores (mean=0, SD=1). ",
      "Q1 (Underserved) counties have high barriers and low AI exposure — highest policy priority. ",
      "Source: FINRA NFCS, CFPB, FCC Form 477."
    )
  ) +
  theme_minimal() +
  theme(
    plot.title    = element_text(face="bold", size=12),
    plot.subtitle = element_text(size=9, color="#555555"),
    plot.caption  = element_text(size=7, color="#888888"),
    legend.position = "bottom"
  )

# Try to add ggrepel labels (optional package)
tryCatch({
  library(ggrepel)
  scatter <- scatter + ggrepel::geom_text_repel(
    data = county_data %>% filter(barrier_severity > 0.5 | ai_exposure > 0.8),
    aes(label = county_name_official),
    size = 2.5, max.overlaps = 15,
    segment.color = "#888888", segment.size = 0.3
  )
}, error = function(e) {
  # ggrepel not installed — add simple labels instead
  scatter <<- scatter + geom_text(
    data = county_data %>% filter(barrier_severity > 0.8),
    aes(label = county_name_official),
    size = 2, hjust = -0.1
  )
})

ggsave(file.path(fig_dir, "ai_gap_index_scatter.png"), scatter,
       width = 10, height = 8, dpi = 300)
cat("  Saved: ai_gap_index_scatter.png\n")

# ── 9. AI Gap Index choropleth map ────────────────────────────
cat("Producing AI Gap Index choropleth map...\n")

wi_gap_map <- wi_sf %>%
  left_join(county_data %>% select(county_fips, quadrant, barrier_severity,
                                    ai_exposure, priority),
            by = "county_fips")

map_gap <- ggplot(wi_gap_map) +
  geom_sf(
    aes(fill = quadrant),
    color    = "#ffffff",
    linewidth= 0.4
  ) +
  scale_fill_manual(
    values   = QUADRANT_COLORS,
    na.value = "#dddddd",
    name     = NULL,
    guide    = guide_legend(nrow = 2, byrow = TRUE)
  ) +
  labs(
    title    = "Wisconsin AI Gap Index",
    subtitle = "County classification by behavioral barrier severity and AI financial tool exposure",
    caption  = paste0(
      "Note: Q1 (Underserved) = high behavioral barriers, low AI exposure — highest intervention priority.\n",
      "Q2 (Emerging) = high barriers, growing AI presence. Q3 (Self-Sufficient) = low barriers, low AI.\n",
      "Q4 (Leaders) = low barriers, high AI exposure. Source: FINRA NFCS, CFPB NLP pipeline, FCC Form 477."
    )
  ) +
  theme_void() +
  theme(
    plot.title      = element_text(face="bold", size=14, hjust=0.5,
                                   margin=margin(b=4)),
    plot.subtitle   = element_text(size=9, hjust=0.5, color="#555555",
                                   margin=margin(b=8)),
    plot.caption    = element_text(size=7, color="#888888", hjust=0,
                                   margin=margin(t=8)),
    legend.position = "bottom",
    legend.text     = element_text(size=8),
    legend.key.size = unit(0.5, "cm"),
    plot.background = element_rect(fill="white", color=NA),
    plot.margin     = margin(10, 10, 10, 10)
  )

ggsave(file.path(fig_dir, "map_ai_gap_index.png"), map_gap,
       width = 8, height = 10, dpi = 300)
cat("  Saved: map_ai_gap_index.png\n\n")

# ── 10. Summary statistics by quadrant ───────────────────────
cat("Summary by quadrant:\n")
quadrant_summary <- county_data %>%
  group_by(quadrant) %>%
  summarise(
    n_counties      = n(),
    mean_enroll     = round(mean(enroll_rate, na.rm=TRUE)*100, 1),
    mean_literacy   = round(mean(mean_literacy, na.rm=TRUE), 2),
    mean_impatience = round(mean(mean_impatience, na.rm=TRUE), 2),
    mean_ai_density = round(mean(ai_density, na.rm=TRUE), 4),
    .groups = "drop"
  ) %>%
  arrange(quadrant)

print(quadrant_summary, n=10)

# ── 11. Final output summary ─────────────────────────────────
cat("\n", strrep("=", 60), "\n", sep="")
cat("AI Gap Index complete.\n\n")
cat("Key outputs:\n")
cat("  data/processed/wi_ai_gap_index.parquet\n")
cat("  outputs/tables/ai_gap_index_rankings.csv\n")
cat("  outputs/figures/map_ai_gap_index.png      <- MAIN POLICY MAP\n")
cat("  outputs/figures/ai_gap_index_scatter.png  <- QUADRANT PLOT\n\n")
cat("ArcGIS Pro next steps:\n")
cat("  1. Open ArcGIS Pro and create a new project\n")
cat("  2. Add data/raw/gis/wi_counties.shp as a layer\n")
cat("  3. Join outputs/tables/ai_gap_index_rankings.csv\n")
cat("     via county_fips field\n")
cat("  4. Symbolize by 'quadrant' field using the four colors above\n")
cat("  5. Add labels, legend, north arrow, scale bar\n")
cat("  6. Export at 300 DPI for publication\n")
cat(strrep("=", 60), "\n")
