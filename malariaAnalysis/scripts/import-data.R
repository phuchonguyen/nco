library(causalRisk)
library(readr)

clean_colnames <- function(df) {
  df_names <- names(df)
  df_names_cleaned <- df_names |>
    sub(" *\\[[[:upper:][:digit:]_]*\\]$", "", x = _, perl = TRUE) |>
    gsub(" ", "_", x = _, perl = TRUE) |>
    gsub("[|(),\\-']", "", x = _, perl = TRUE) |>
    gsub("<", "_lt_", x = _, perl = TRUE) |>
    gsub("<", "_lt_", x = _, perl = TRUE) |>
    gsub("<=", "_le_", x = _, perl = TRUE) |>
    gsub(">", "_gt_", x = _, perl = TRUE) |>
    gsub(">=", "_ge_", x = _, perl = TRUE) |>
    gsub("__+", "_", x = _, perl = TRUE) |>
    tolower() |>
    make.names(TRUE)
  names(df) <- df_names_cleaned
  df
}

ps <- clean_colnames(read_tsv("../data/PROMO-1_Participant_subsettedData.txt"))
hh <- clean_colnames(read_tsv("../data/PROMOTE_BC3_cohort_Households.txt"))
prm <- clean_colnames(read_tsv("../data/PROMO-1_Participant repeated measure_subsettedData.txt"))

promote <- dplyr::left_join(ps, hh, by = "household_id")

outcomes <- c(
  "hypertension",
  "diabetes_mellitus",
  "rheumatic_fever",
  "cardiac_disease",
  "renal_disease",
  "asthma",
  "sickle_cell_disease",
  "abdominal_surgery",                        #  34
  "blood_transfusion",                        #  15
  "pelvis_spine_or_femur_fracture",
  "major_injury_from_road_traffic_accident",
  "familial_diabetes",                        #  34
  "familial_hypertension",                    #  70
  "familial_multiple_gestation",              # 191
  "drug_allergies"
)

covariates <- c(
  "alcohol_use",
  # "education_level",  # need to collapse
  "age_at_enrollment_years",
  "bank_account",
  "bed",
  "bicycle",
  # "car_or_truck",
  # "drinking_water_source",  # need to collapse
  # "dwelling_type",  # perfect correlation with roof_material_type
  "household_wealth_index_categorical",
  "latrine_type",
  "meals_per_day_categorization"
  # "transit_to_health_facility"  # need to collapse
)

treatments <- c(
  "roof_material_type",
  "landline_phone",
  "wall_material_type"
)
