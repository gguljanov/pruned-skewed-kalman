# eval_p <- matrix(c(0.2, 0.3, -0.8), ncol = 1)
# Corr_mat <- matrix(c(1.0, 0.2, 0.3, -0.4, 1.0, -0.9, -0.5, 0.8, 1.0),
#     ncol = 3, byrow = TRUE
# )
# Corr_mat <- 0.5 * (Corr_mat + t(Corr_mat))
# print(logcdf_ME(eval_p, Corr_mat))

library(R.matlab)

data <- readMat("../../Skewed_lin_DSGE/Codes/test_data.mat")
data <- t(data$y.mat)


source("skalman_filter.R")
# res <- kalman_csn(
#     Y = data,
#     mu_tm1_tm1 = matrix(0),
#     Sigma_tm1_tm1 = matrix(10),
#     Gamma_tm1_tm1 = matrix(0),
#     nu_tm1_tm1 = matrix(0),
#     Delta_tm1_tm1 = matrix(1),
#     G = matrix(0.5),
#     R = matrix(1),
#     F = matrix(0.3),
#     mu_eta = matrix(0),
#     Sigma_eta = matrix(1),
#     Gamma_eta = matrix(0.5),
#     nu_eta = matrix(0),
#     Delta_eta = matrix(1),
#     mu_eps = matrix(0),
#     Sigma_eps = matrix(1),
#     cut_tol = 1e-1,
#     eval_lik = TRUE,
#     ret_pred_filt = FALSE,
#     logcdfmvna_fct = logcdf_ME
# )

source("skalman_filter.R")
res <- kalman_csn(
    Y = data,
    mu_tm1_tm1 = diag(0, 2, 1),
    Sigma_tm1_tm1 = matrix(c(10, 0, 0, 10), ncol = 2),
    Gamma_tm1_tm1 = diag(0, 2, 2),
    nu_tm1_tm1 = diag(0, 2, 1),
    Delta_tm1_tm1 = diag(1, 2, 2),
    G = matrix(c(0.7, -0.5, 0.3, 0.9), ncol = 2, byrow = TRUE),
    R = diag(1, 2, 2),
    F = matrix(c(0.3, 1, 2, -5), ncol = 2, byrow = TRUE),
    mu_eta = diag(0, 2, 1),
    Sigma_eta = diag(1, 2, 2),
    Gamma_eta = matrix(c(0.5, 1.2, -1.8, 2), ncol = 2, byrow = TRUE),
    nu_eta = diag(0, 2, 1),
    Delta_eta = diag(1, 2, 2),
    mu_eps = diag(0, 2, 1),
    Sigma_eps = diag(1, 2, 2),
    cut_tol = 1e-1,
    eval_lik = TRUE,
    ret_pred_filt = TRUE,
    logcdfmvna_fct = logcdf_ME
)

print(res$log_lik)
print(res$pred$nu[[30]])
print(res$filt$nu[[30]])
