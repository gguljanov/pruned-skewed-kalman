using Debugger
using LinearAlgebra
using Distributions
using Random
using Optim
include("skalman_filter.jl")

# Simulate data
y_nbr = 2
obs_nbr = 1000

G_mat = [0.7 -0.5; 0.3 0.9]
R_mat = Matrix(1.0I, 2, 2)
F_mat = [0.3 1; 2 -5]

mu_eta = Matrix(0.0I, 2, 1)
Sigma_eta = Matrix(1.0I, 2, 2)

Gamma_eta = [0.5 1.2; -1.8 2]
nu_eta = Matrix(0.0I, 2, 1)
Delta_eta = Matrix(1.0I, 2, 2)
mu_eps = Matrix(0.0I, 2, 1)
Sigma_eps = Matrix(1.0I, 2, 2)

# Simulate the data
state_X = Matrix(0.0I, y_nbr, 1)
data_Y = Array{Float64, 2}(undef, y_nbr, obs_nbr)

eta_vec = csnRandDirect(
    nn=obs_nbr,
    mu=mu_eta,
    Sigma=Sigma_eta,
    Gamma=Gamma_eta,
    nu=nu_eta,
    Delta=Delta_eta
)

StandardNormalDist = Normal()

eps_vec = standard_norm_sample = (
    mu_eps .+ cholesky(Sigma_eps).L * rand(StandardNormalDist, y_nbr, obs_nbr)
)

for ii in 1:obs_nbr 
    state_X = G_mat * state_X + R_mat * eta_vec[:, ii]
    data_Y[:, ii] = F_mat * state_X + eps_vec[:, ii]
end


# Function to be given to the optimization procedure
function neg_log_likeli(param)
    Gamma_eta = [param[1] param[2]; -1.8 2]

    log_lik, pred, filt = kalman_csn(
        Y=data_Y,
        mu_tm1_tm1=zeros(2),
        Sigma_tm1_tm1=[10.00 0.00; 0.00 10.00],
        Gamma_tm1_tm1=zeros(2, 2),
        nu_tm1_tm1=zeros(2),
        Delta_tm1_tm1=Matrix(1.00I, 2, 2),
        G=[0.70 -0.50; 0.3 0.9],
        R=Matrix(1.00I, 2, 2),
        F=[0.30 1.00; 2.00 -5.00],
        mu_eta=zeros(2),
        Sigma_eta=Matrix(1.00I, 2, 2),
        Gamma_eta=Gamma_eta,
        nu_eta=zeros(2),
        Delta_eta=Matrix(1.00I, 2, 2),
        mu_eps=zeros(2),
        Sigma_eps=Matrix(1.00I, 2, 2),
        cut_tol=0.1,
        eval_lik=true,
        ret_pred_filt=false,
        logcdfmvna_fct=logcdf_ME,
    )

    # Negative log-likelihood
    return(-1 * log_lik)
end

param0 = [0.5 1.2]
neg_log_likeli(param0)

# Minimize the negative log-likelihood to estimate the parameter
OptimRes = optimize(neg_log_likeli, param0, NelderMead())

Optim.minimizer(OptimRes)
