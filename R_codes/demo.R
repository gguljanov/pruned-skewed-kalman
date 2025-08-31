# In this script, we will demonstrate how the Pruned Skewed Kalman filter
# works, by first simulating a dataset using a skewed state space model
# and then re-estimating the model

library(csn)
source("skalman_filter.R")

# Set the parameter values
y_nbr <- 2
obs_nbr <- 1e3

G_mat <- matrix(c(0.7, -0.5, 0.3, 0.9), ncol = y_nbr, byrow = TRUE)
R_mat <- diag(1, 2, 2)
F_mat <- matrix(c(0.3, 1, 2, -5), ncol = y_nbr, byrow = TRUE)

mu_eta <- diag(0, 2, 1)
Sigma_eta <- diag(1, 2, 2)

Gamma_eta <- matrix(c(0.5, 1.2, -1.8, 2), ncol = y_nbr, byrow = TRUE)
nu_eta <- diag(0, 2, 1)
Delta_eta <- diag(1, 2, 2)
mu_eps <- diag(0, 2, 1)
Sigma_eps <- diag(1, 2, 2)

# Simulate the data
state_X <- matrix(0, ncol = 1, nrow = y_nbr)
data_Y <- matrix(NA, ncol = obs_nbr, nrow = y_nbr)

eta_vec <- rcsn(
    k = obs_nbr,
    mu = drop(mu_eta),
    sigma = Sigma_eta,
    gamma = Gamma_eta,
    nu = drop(nu_eta),
    delta = Delta_eta
)

eta_vec <- t(eta_vec)

standard_norm_sample <- matrix(
    data = rnorm(n = obs_nbr, mean = mu_eps, sd = sqrt(Sigma_eps)),
    nrow = y_nbr,
    ncol = obs_nbr
)

eps_vec <- drop(mu_eps) + chol(Sigma_eps) %*% standard_norm_sample

for (ii in seq(obs_nbr)) {
    state_X <- G_mat %*% state_X + R_mat %*% eta_vec[, ii, drop = FALSE]
    data_Y[, ii] <- F_mat %*% state_X + eps_vec[, ii, drop = FALSE]
}

# Function to be given to the optimization procedure
neg_log_likeli <- function(param) {
    Gamma_eta <- matrix(c(param[1], param[2], -1.8, 2), ncol = 2, byrow = TRUE)

    filter_res <- kalman_csn(
        Y = data_Y,
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
        Gamma_eta = Gamma_eta,
        nu_eta = diag(0, 2, 1),
        Delta_eta = diag(1, 2, 2),
        mu_eps = diag(0, 2, 1),
        Sigma_eps = diag(1, 2, 2),
        cut_tol = 1e-1,
        eval_lik = TRUE,
        ret_pred_filt = FALSE,
        logcdfmvna_fct = logcdf_ME
    )

    # Negative log-likelihood
    return(-1 * filter_res$log_lik)
}

print(neg_log_likeli(c(0.5, 1.2)))

# Minimize the negative log-likelihood to estimate the parameter
optim_res <- optim(
    par = c(-0.5, -1.2),
    fn = neg_log_likeli,
    method = "Nelder-Mead"
)

print(optim_res)
