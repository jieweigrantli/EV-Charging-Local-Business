#!/usr/bin/env Rscript
# crosscheck_two_part_model.R
# ---------------------------
# Independent re-estimation of the two-part hurdle model (Table 2 /
# Supplementary Table 2) in R, using the same analytic dataset and the same
# specification as code/02_two_part_model.py.  HC3 standard errors are
# computed with the sandwich package.  Writes
#   outputs/tables/r_crosscheck_two_part_model.csv
# and prints the maximum absolute difference to the Python estimates when the
# Python output (outputs/tables/supplementary_table_2_full_model.csv) exists.
#
# Requires R >= 4.1 with packages sandwich and lmtest.

suppressPackageStartupMessages({
  library(sandwich)
  library(lmtest)
})

root <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1])), "..", ".."))
data_file <- Sys.getenv("EVCLB_DATA", unset = file.path(root, "data", "filtered_dataset.csv"))
if (!file.exists(data_file)) data_file <- file.path(root, "data", "filtered_dataset_public.csv")
df <- read.csv(data_file, check.names = FALSE)
cat("Data:", basename(data_file), nrow(df), "sessions\n")

predictors <- c(
  "race_asian", "race_black", "race_hispanic", "gender_female", "edu_college", "edu_grad",
  "hh_adults", "hh_children", "charge_duration", "wait_time",
  "250m_department_store", "250m_entertainment", "250m_grocery", "250m_medical", "250m_park",
  "250m_religion", "250m_restaurant", "250m_standalone_bathroom", "250m_colocated_bathroom",
  "250m_retail", "250m_work",
  "veh_use_commuting", "veh_use_private", "veh_use_business", "veh_use_gigecon",
  "charge_session_reason_long_trip", "charge_session_reason_opportunity",
  "battery_capacity_exist", "distance_from_home_exist", "log_income_continuous_exist",
  "age_continuous_exist", "battery_capacity_centered", "distance_from_home_log1p_centered",
  "session_cost_log1p_centered", "log_income_centered", "age_centered")
missing <- setdiff(predictors, names(df))
if (length(missing)) {
  message("Predictors not in dataset (public version?) and omitted: ", paste(missing, collapse = ", "))
  predictors <- setdiff(predictors, missing)
}
target <- "store_total_spent_calc"

X <- df[, predictors]
X[] <- lapply(X, function(v) suppressWarnings(as.numeric(v)))
complete <- stats::complete.cases(X)
d <- X[complete, ]
y <- suppressWarnings(as.numeric(df[[target]][complete]))
d$spend_any <- as.integer(!is.na(y) & y > 0)
names(d) <- make.names(names(d))          # 250m_* -> X250m_*

glm_fit <- glm(spend_any ~ ., data = d, family = binomial(link = "logit"))
glm_hc3 <- coeftest(glm_fit, vcov. = vcovHC(glm_fit, type = "HC3"))

pos <- !is.na(y) & y > 0
d2 <- d[pos, setdiff(names(d), "spend_any")]
d2$log_spend <- log1p(y[pos])
ols_fit <- lm(log_spend ~ ., data = d2)
ols_hc3 <- coeftest(ols_fit, vcov. = vcovHC(ols_fit, type = "HC3"))

cat(sprintf("Extensive margin: N = %d, log-lik = %.2f\n", nobs(glm_fit), as.numeric(logLik(glm_fit))))
cat(sprintf("Intensive margin: N = %d, R2 = %.3f, log-lik = %.2f\n", nobs(ols_fit), summary(ols_fit)$r.squared,
            as.numeric(logLik(ols_fit))))

tidy <- function(ct, model) {
  data.frame(model = model, term = rownames(ct), coef = ct[, 1], se = ct[, 2], p_two_sided = ct[, 4],
             row.names = NULL)
}
out <- rbind(tidy(glm_hc3, "extensive_logit"), tidy(ols_hc3, "intensive_ols"))
out$term <- sub("^X250m_", "250m_", out$term)
out$term[out$term == "(Intercept)"] <- "const"
out_file <- file.path(root, "outputs", "tables", "r_crosscheck_two_part_model.csv")
dir.create(dirname(out_file), showWarnings = FALSE, recursive = TRUE)
write.csv(out, out_file, row.names = FALSE)
cat("Wrote", out_file, "\n")

py_file <- file.path(root, "outputs", "tables", "supplementary_table_2_full_model.csv")
if (file.exists(py_file)) {
  py <- read.csv(py_file, check.names = FALSE)
  g <- merge(out[out$model == "extensive_logit", ], py[, c("term", "coef_logit", "se_logit")], by = "term")
  o <- merge(out[out$model == "intensive_ols", ], py[, c("term", "coef_ols", "se_ols")], by = "term")
  cat(sprintf("Max |coef difference| vs Python: logit %.2e, OLS %.2e\n",
              max(abs(g$coef - g$coef_logit)), max(abs(o$coef - o$coef_ols))))
  cat(sprintf("Max |HC3 SE difference| vs Python: logit %.2e, OLS %.2e\n",
              max(abs(g$se - g$se_logit)), max(abs(o$se - o$se_ols))))
}
cat("R", R.version$major, ".", R.version$minor, "; sandwich ", as.character(packageVersion("sandwich")),
    "; lmtest ", as.character(packageVersion("lmtest")), "\n", sep = "")
