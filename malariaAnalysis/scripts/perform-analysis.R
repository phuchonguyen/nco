library(dplyr)
# library(purrr)
set.seed(76543)

# Note that the call to `sort` removes `NA_character_` values
sample_missing <- function(x) {
  stopifnot(is.character(x))
  counts <- table(x)
  prob <- counts / sum(counts)
  if_else(is.na(x), sample(names(counts), length(x), TRUE, prob), x)
}

# Note that the call to `sort` removes `NA_character_` values
to_event <- function(x) {
  stopifnot(is.character(x))
  smallest_level <- names(sort(table(x))[1L])
  if_else(
    x == smallest_level,
    1,
    NA_real_
  )
}

to_censoring <- function(x) {
  if_else(is.na(x), 2, NA_real_)
}

promote_mung <- promote |>
  filter(complete.cases(promote[, c(outcomes, covariates, treatments)])) |>
  mutate(
    roof_material_type_imputed = sample_missing(roof_material_type),
    landline_phone_imputed = sample_missing(landline_phone),
    wall_material_type_imputed = sample_missing(wall_material_type),
    familial_multiple_gestation_tte = to_event(familial_multiple_gestation),
    familial_multiple_gestation_censoring = to_censoring(familial_multiple_gestation_tte)
  ) |>
  modify_if(is.character, as.factor)


models_covs_no = specify_models(
  identify_treatment(roof_material_type_imputed),
  identify_censoring(familial_multiple_gestation_censoring),
  identify_outcome(familial_multiple_gestation_tte)
)

trt <- as.formula(sprintf("~ %s", paste0(covariates, collapse = " + ")))
models_covs_yes = specify_models(
  identify_treatment(
    roof_material_type_imputed,
    trt
  ),
  identify_censoring(familial_multiple_gestation_censoring),
  identify_outcome(familial_multiple_gestation_tte)
)

fit_covs_no = estimate_ipwrisk(
  promote_mung,
  models_covs_no,
  labels = c("Overall cumulative risk")
)

fit_covs_yes = estimate_ipwrisk(
  promote_mung,
  models_covs_yes,
  labels = c("Overall cumulative risk")
)
