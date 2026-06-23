# Merge Log — Wisconsin Retirement AI
Generated: 2026-06-22 18:26:21
Final dataset: 1544 rows x 50 columns

## Merge Steps

### Step 1: FCC Broadband (IV)
- Rows before:  1,544
- Rows after:   1,544
- Unmatched:    0
- Merge key:    county_fips + SURVEY_WAVE

### Step 2: ACS County Controls
- Rows before:  1,544
- Rows after:   1,544
- Unmatched:    0
- Merge key:    county_fips + SURVEY_WAVE->nfcs_wave

### Step 3: BLS Unemployment Rate
- Rows before:  1,544
- Rows after:   1,544
- Unmatched:    0
- Merge key:    county_fips + SURVEY_WAVE

## Variables in Master Dataset
AGE_NUM, ai_interest, bb_providers_100_10, bb_providers_25_3, bias_cc_revolve, bias_overdraft, bias_payday, county_fips, education, employment, enrolled, era, female, fin_ed_received, fin_ed_school, fin_fragility, homeowner, impatience_index, impatience_std, income, literacy_score, literacy_std, log_bb_25_3, log_median_income, low_income, low_literacy, marital, median_hh_income, minority, NFCSID, pct_asian_nh, pct_bach_plus, pct_black_nh, pct_broadband, pct_hispanic, pct_owner_occ, pct_white_nh, pop_total, poverty_rate, race_eth, race_eth_restricted, ret_calc, STATEQ_NUM, SURVEY_WAVE, unemp_rate, wave_fe, wave_year, weight_nat, weight_st, zip_clean

## Next Step
Run Phase 2 NLP pipeline (code/02_nlp_pipeline/run_pipeline.py)
Then run 01h_master_merge_nlp.R to add NLP outputs to master dataset.
