# ============================================================
# Phase 4 — Script 04a: Probit Models (Main Results)
# ============================================================
# Project: Wisconsin Retirement AI
# Script:  code/04_econometrics/04a_probit_models.R
# Author:  Khawaja Sazzad Ali
# Date:    Summer 2026
#
# WHAT THIS SCRIPT DOES:
#   Estimates the four core Probit models and reports average
#   marginal effects with county-clustered standard errors.
#   This is the primary results table of the paper.
#
# MODELS:
#   Model 1: Behavioral baseline (literacy + impatience + controls)
#   Model 2: AI moderation (+ AI density + Literacy x AI)
#   Model 3: Behavioral equalizer (+ Impatience x AI)
#   Model 4: Heterogeneity (+ triple interactions)
#
# ALL MODELS INCLUDE:
#   - Wave fixed effects
#   - County-clustered standard errors
#   - Average marginal effects reported
#
# INPUT:  data/final/master_analytical.parquet
# OUTPUT: outputs/tables/table2_probit_main.csv
#         outputs/tables/table2_probit_main.tex
#         outputs/figures/fig_marginal_effects.png
#         docs/phase4_log.txt
# ============================================================

library(tidyverse)
library(arrow)
library(here)
library(glue)
library(fixest)        # Fast Probit with clustered SE and FE
library(modelsummary)  # Publication-quality tables
library(marginaleffects) # Average marginal effects

# ── 0. Setup ─────────────────────────────────────────────────
final_dir <- here("data", "final")
tbl_dir   <- here("outputs", "tables")
fig_dir   <- here("outputs", "figures")
docs_dir  <- here("docs")
dir.create(tbl_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

log_lines <- c()
log <- function(msg) { cat(msg, "\n"); log_lines <<- c(log_lines, msg) }

log(strrep("=", 60))
log("Phase 4 — Probit Models (Main Results)")
log(format(Sys.time()))
log(strrep("=", 60))

# ── 1. Load analytical dataset ────────────────────────────────
log("\n[1] Loading analytical dataset...")
df <- read_parquet(file.path(final_dir, "master_analytical.parquet"))
log(glue("  Rows: {format(nrow(df), big.mark=',')} | Cols: {ncol(df)}"))

# Ensure factor types
df <- df %>%
  mutate(
    wave_fe     = factor(SURVEY_WAVE),
    county_fips = as.character(county_fips),
    enrolled    = as.integer(enrolled),
    female      = as.integer(female),
    minority    = as.integer(minority),
    low_income  = as.integer(low_income)
  )


# ── 2. Define model samples ───────────────────────────────────
log("\n[2] Defining model samples...")

# Model 1 sample: all respondents with complete core variables
df_m1 <- df %>%
  filter(
    !is.na(enrolled),
    !is.na(literacy_score),
    !is.na(impatience_index),
    !is.na(log_median_income),
    !is.na(poverty_rate),
    !is.na(pct_bach_plus),
    !is.na(unemp_rate)
  )
log(glue("  Model 1 sample: {format(nrow(df_m1), big.mark=',')} obs"))

# Models 2-4 sample: also need AI density and distress index
df_m24 <- df_m1 %>%
  filter(
    !is.na(ai_density_log),
    !is.na(distress_index)
  )
log(glue("  Models 2-4 sample: {format(nrow(df_m24), big.mark=',')} obs"))

# Model 4 female subsample
df_m4f <- df_m24 %>% filter(!is.na(female))
log(glue("  Model 4 female subsample: {format(nrow(df_m4f), big.mark=',')} obs"))


# ── 3. Common controls ────────────────────────────────────────
# These enter all four models identically
CONTROLS <- c(
  "log_median_income",
  "poverty_rate",
  "pct_bach_plus",
  "unemp_rate"
)


# ── 4. Model 1 — Behavioral Baseline ─────────────────────────
log("\n[3] Estimating models...")
log("  Model 1: Behavioral baseline...")

m1 <- feglm(
  enrolled ~
    literacy_score +
    impatience_index +
    log_median_income +
    poverty_rate +
    pct_bach_plus +
    unemp_rate |
    wave_fe,                        # Wave fixed effects
  data   = df_m1,
  family = binomial("probit"),
  cluster = ~county_fips            # County-clustered SE
)

log(glue("    N = {m1$nobs} | Converged: {m1$convStatus == 0}"))


# ── 5. Model 2 — AI Moderation ───────────────────────────────
log("  Model 2: AI moderation (Literacy x AI)...")

m2 <- feglm(
  enrolled ~
    literacy_score +
    impatience_index +
    ai_density_log +
    lit_x_ai +
    distress_index +
    log_median_income +
    poverty_rate +
    pct_bach_plus +
    unemp_rate |
    wave_fe,
  data    = df_m24,
  family  = binomial("probit"),
  cluster = ~county_fips
)

log(glue("    N = {m2$nobs} | Converged: {m2$convStatus == 0}"))


# ── 6. Model 3 — Behavioral Equalizer ────────────────────────
log("  Model 3: Behavioral equalizer (+ Impatience x AI)...")

m3 <- feglm(
  enrolled ~
    literacy_score +
    impatience_index +
    ai_density_log +
    lit_x_ai +
    imp_x_ai +
    distress_index +
    log_median_income +
    poverty_rate +
    pct_bach_plus +
    unemp_rate |
    wave_fe,
  data    = df_m24,
  family  = binomial("probit"),
  cluster = ~county_fips
)

log(glue("    N = {m3$nobs} | Converged: {m3$convStatus == 0}"))


# ── 7. Model 4 — Heterogeneity ───────────────────────────────
log("  Model 4: Heterogeneity (triple interactions)...")

# Check which triple interactions have enough variation
n_lowinc_ai <- sum(!is.na(df_m24$lit_x_ai_x_lowinc) &
                    df_m24$low_income == 1, na.rm=TRUE)
n_minority_ai <- sum(!is.na(df_m24$imp_x_ai_x_minority) &
                      df_m24$minority == 1, na.rm=TRUE)
n_female_ai  <- sum(!is.na(df_m4f$imp_x_ai_x_female) &
                     df_m4f$female == 1, na.rm=TRUE)

log(glue("    Low income x AI subsample: {n_lowinc_ai}"))
log(glue("    Minority x AI subsample: {n_minority_ai}"))
log(glue("    Female x AI subsample: {n_female_ai}"))

# Run Model 4 with low_income and minority triple interactions
# Female interaction uses restricted female subsample
m4 <- feglm(
  enrolled ~
    literacy_score +
    impatience_index +
    ai_density_log +
    lit_x_ai +
    imp_x_ai +
    lit_x_ai_x_lowinc +
    imp_x_ai_x_minority +
    low_income +
    minority +
    distress_index +
    log_median_income +
    poverty_rate +
    pct_bach_plus +
    unemp_rate |
    wave_fe,
  data    = df_m24,
  family  = binomial("probit"),
  cluster = ~county_fips
)

log(glue("    N = {m4$nobs} | Converged: {m4$convStatus == 0}"))

# Model 4b — Female triple interaction (separate subsample)
if (nrow(df_m4f) >= 100 && n_female_ai >= 30) {
  m4b <- tryCatch({
    feglm(
      enrolled ~
        literacy_score +
        impatience_index +
        ai_density_log +
        lit_x_ai +
        imp_x_ai +
        imp_x_ai_x_female +
        female +
        distress_index +
        log_median_income +
        poverty_rate +
        pct_bach_plus +
        unemp_rate |
        wave_fe,
      data    = df_m4f,
      family  = binomial("probit"),
      cluster = ~county_fips
    )
  }, error = function(e) {
    log(glue("    Model 4b (female) failed: {e$message}"))
    NULL
  })
  if (!is.null(m4b))
    log(glue("    Model 4b (female) N = {m4b$nobs}"))
} else {
  m4b <- NULL
  log("    Model 4b (female) skipped — insufficient subsample")
}


# ── 8. Average marginal effects ───────────────────────────────
log("\n[4] Computing average marginal effects (AME)...")

compute_ame <- function(model, label) {
  tryCatch({
    ame <- avg_slopes(model)
    log(glue("  {label}: {nrow(ame)} effects computed"))
    ame$model <- label
    return(ame)
  }, error = function(e) {
    log(glue("  {label} AME failed: {e$message}"))
    return(NULL)
  })
}

ame_m1 <- compute_ame(m1, "Model 1")
ame_m2 <- compute_ame(m2, "Model 2")
ame_m3 <- compute_ame(m3, "Model 3")
ame_m4 <- compute_ame(m4, "Model 4")


# ── 9. Key results summary ────────────────────────────────────
log("\n[5] Key coefficient results:")
log(strrep("-", 60))

summarize_coef <- function(model, varnames, label) {
  log(glue("\n  {label}:"))
  coefs <- coef(model)
  ses   <- sqrt(diag(vcov(model)))
  for (v in varnames) {
    if (v %in% names(coefs)) {
      b  <- round(coefs[v], 4)
      se <- round(ses[v], 4)
      t  <- round(b / se, 2)
      sig <- case_when(
        abs(t) > 2.576 ~ "***",
        abs(t) > 1.960 ~ "**",
        abs(t) > 1.645 ~ "*",
        TRUE           ~ ""
      )
      log(glue("    {str_pad(v, 25)}: b={b}, SE={se}, t={t}{sig}"))
    }
  }
}

summarize_coef(m1, c("literacy_score", "impatience_index"),
               "Model 1 — Behavioral baseline")
summarize_coef(m2, c("literacy_score", "impatience_index",
                      "ai_density_log", "lit_x_ai"),
               "Model 2 — AI moderation")
summarize_coef(m3, c("literacy_score", "impatience_index",
                      "ai_density_log", "lit_x_ai", "imp_x_ai"),
               "Model 3 — Behavioral equalizer")
summarize_coef(m4, c("ai_density_log", "lit_x_ai",
                      "imp_x_ai", "lit_x_ai_x_lowinc",
                      "imp_x_ai_x_minority"),
               "Model 4 — Heterogeneity")


# ── 10. AME summary table ─────────────────────────────────────
log("\n[6] Average Marginal Effects — key variables:")
log(strrep("-", 60))

print_ame <- function(ame_df, vars, label) {
  if (is.null(ame_df)) return(invisible())
  log(glue("\n  {label}:"))
  ame_sub <- ame_df %>%
    filter(term %in% vars) %>%
    select(term, estimate, std.error, statistic, p.value) %>%
    mutate(
      across(c(estimate, std.error, statistic), ~ round(.x, 4)),
      p.value = round(p.value, 4),
      sig = case_when(
        p.value < 0.01 ~ "***",
        p.value < 0.05 ~ "**",
        p.value < 0.10 ~ "*",
        TRUE           ~ ""
      )
    )
  print(ame_sub, n=20)
}

key_vars <- c("literacy_score", "impatience_index",
               "ai_density_log", "lit_x_ai", "imp_x_ai")

print_ame(ame_m1, c("literacy_score", "impatience_index"),
          "Model 1 AME")
print_ame(ame_m2, key_vars, "Model 2 AME")
print_ame(ame_m3, key_vars, "Model 3 AME")


# ── 11. Publication table ─────────────────────────────────────
log("\n[7] Producing publication table...")

model_list <- list(
  "Model 1\nBaseline"    = m1,
  "Model 2\nAI Mod."     = m2,
  "Model 3\nEqualizer"   = m3,
  "Model 4\nHeterog."    = m4
)

coef_map <- c(
  "literacy_score"         = "Financial literacy score",
  "impatience_index"       = "Impatience index",
  "ai_density_log"         = "Log AI complaint density",
  "lit_x_ai"               = "Literacy × AI density",
  "imp_x_ai"               = "Impatience × AI density",
  "lit_x_ai_x_lowinc"      = "Literacy × AI × Low income",
  "imp_x_ai_x_minority"    = "Impatience × AI × Minority",
  "low_income"             = "Low income (0/1)",
  "minority"               = "Minority (0/1)",
  "distress_index"         = "Financial distress index",
  "log_median_income"      = "Log median income",
  "poverty_rate"           = "County poverty rate",
  "pct_bach_plus"          = "County % bachelor's+",
  "unemp_rate"             = "County unemployment rate"
)

gof_map <- tribble(
  ~raw,           ~clean,          ~fmt,
  "nobs",         "Observations",   0,
  "pseudo_r2",    "Pseudo R²",      3,
  "aic",          "AIC",            1
)

tryCatch({
  tbl_probit <- modelsummary(
    model_list,
    coef_map   = coef_map,
    gof_map    = gof_map,
    stars      = c("*" = 0.1, "**" = 0.05, "***" = 0.01),
    title      = "Table 2: Probit Models of Retirement Plan Enrollment",
    notes      = paste(
      "Notes: Probit coefficients reported.",
      "Standard errors clustered at county level in parentheses.",
      "All models include survey wave fixed effects.",
      "* p<0.10, ** p<0.05, *** p<0.01"
    ),
    output     = "dataframe"
  )

  csv_path <- file.path(tbl_dir, "table2_probit_main.csv")
  write_csv(tbl_probit, csv_path)
  log(glue("  Table saved: {csv_path}"))

}, error = function(e) {
  log(glue("  modelsummary failed: {e$message}"))
  log("  Saving raw coefficient table instead...")

  # Fallback: manual coefficient extraction
  extract_coefs <- function(model, name) {
    coefs <- coef(model)
    ses   <- sqrt(diag(vcov(model)))
    tibble(
      term  = names(coefs),
      !!name := paste0(round(coefs, 4), "\n(",
                       round(ses, 4), ")")
    )
  }

  raw_tbl <- reduce(
    list(
      extract_coefs(m1, "Model_1"),
      extract_coefs(m2, "Model_2"),
      extract_coefs(m3, "Model_3"),
      extract_coefs(m4, "Model_4")
    ),
    full_join, by = "term"
  )

  csv_path <- file.path(tbl_dir, "table2_probit_main.csv")
  write_csv(raw_tbl, csv_path)
  log(glue("  Fallback table saved: {csv_path}"))
})


# ── 12. Marginal effects plot ─────────────────────────────────
log("\n[8] Producing marginal effects plot...")

tryCatch({
  # Combine AMEs from models 1-3 for key behavioral variables
  ame_combined <- bind_rows(
    ame_m1 %>% mutate(model = "Model 1\nBaseline"),
    ame_m2 %>% mutate(model = "Model 2\nAI Moderation"),
    ame_m3 %>% mutate(model = "Model 3\nEqualizer")
  ) %>%
    filter(term %in% c("literacy_score", "impatience_index",
                        "ai_density_log", "lit_x_ai", "imp_x_ai")) %>%
    mutate(
      term_label = recode(term,
        literacy_score   = "Literacy score",
        impatience_index = "Impatience index",
        ai_density_log   = "Log AI density",
        lit_x_ai         = "Literacy × AI",
        imp_x_ai         = "Impatience × AI"
      ),
      sig = case_when(
        p.value < 0.01 ~ "p<0.01",
        p.value < 0.05 ~ "p<0.05",
        p.value < 0.10 ~ "p<0.10",
        TRUE           ~ "n.s."
      )
    )

  p_ame <- ggplot(
    ame_combined,
    aes(x = estimate, y = term_label, color = model, shape = sig)
  ) +
    geom_vline(xintercept = 0, linetype = "dashed",
               color = "#666666", linewidth = 0.5) +
    geom_errorbarh(
      aes(xmin = conf.low, xmax = conf.high),
      height = 0.2, linewidth = 0.6, alpha = 0.7
    ) +
    geom_point(size = 3) +
    facet_wrap(~ model, ncol = 3) +
    scale_color_manual(
      values = c("#2166ac", "#d6604d", "#4dac26"),
      guide  = "none"
    ) +
    scale_shape_manual(
      values = c("p<0.01"=16, "p<0.05"=17, "p<0.10"=15, "n.s."=1),
      name   = "Significance"
    ) +
    labs(
      title   = "Figure 2: Average Marginal Effects on Retirement Plan Enrollment",
      subtitle= "Probit models with county-clustered standard errors and wave fixed effects",
      x       = "Average marginal effect (percentage points)",
      y       = NULL,
      caption = paste(
        "Note: Points show average marginal effects; bars show 95% confidence intervals.",
        "Models estimated on Wisconsin NFCS respondents (pooled 2009-2024).",
        "Models 2-3 use AI-subsample (n=867); Model 1 uses full sample (n=1,544)."
      )
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title    = element_text(face = "bold", size = 12),
      plot.subtitle = element_text(size = 9, color = "#555555"),
      plot.caption  = element_text(size = 7, color = "#888888"),
      legend.position = "bottom",
      strip.text    = element_text(face = "bold")
    )

  fig_path <- file.path(fig_dir, "fig_marginal_effects.png")
  ggsave(fig_path, p_ame, width = 12, height = 6, dpi = 300)
  log(glue("  Figure saved: {fig_path}"))

}, error = function(e) {
  log(glue("  Plot failed: {e$message}"))
})


# ── 13. Save models and log ───────────────────────────────────
# Save model objects for Phase 5 (AI Gap Index uses residuals)
saveRDS(
  list(m1=m1, m2=m2, m3=m3, m4=m4),
  file.path(final_dir, "probit_models.rds")
)
log(glue("\nModel objects saved: data/final/probit_models.rds"))

log_path <- file.path(docs_dir, "phase4_probit_log.txt")
writeLines(log_lines, log_path)
log(glue("Log saved: {log_path}"))

log(strrep("=", 60))
log("Script 04a complete.")
log("Next: Run 04b_iv_probit.R for IV-Probit specifications")
log(strrep("=", 60))
