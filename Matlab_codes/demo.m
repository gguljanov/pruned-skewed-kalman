% Simulate data
y_nbr = 2;
obs_nbr = 1000;

G_mat = [0.7, -0.5; 0.3, 0.9];
R_mat = eye(2);
F_mat = [0.3, 1; 2, -5];

mu_eta = ones(2, 1);
Sigma_eta = eye(2);

Gamma_eta = [0.5, 1.2; -1.8, 2];
nu_eta = ones(2, 1);
Delta_eta = eye(2);
mu_eps = zeros(2, 1);
Sigma_eps = eye(2);

% Simulate the data
state_X = ones(2, 1);
data_Y = zeros(y_nbr, obs_nbr);

eta_vec = csnRandDirect(
    obs_nbr, mu_eta, Sigma_eta, Gamma_eta, nu_eta, Delta_eta
);

eps_vec = mu_eps + chol(Sigma_eps, "lower") * randn(y_nbr, obs_nbr);

for ii = 1:obs_nbr 
    state_X = G_mat * state_X + R_mat * eta_vec(:, ii); 
    data_Y(:, ii) = F_mat * state_X + eps_vec(:, ii);
end


% Function to be given to the optimization procedure
mu_tm1_tm1=zeros(2, 1);
Sigma_tm1_tm1=[10, 0; 0, 10];
Gamma_tm1_tm1=zeros(2, 2);
nu_tm1_tm1=zeros(2, 1);
Delta_tm1_tm1=eye(2);
logcdfmvna_fct="logcdf_ME";
cut_tol=0.1;
skip_lik=false;
skip_loss=true;% Simulate data
y_nbr = 2;
obs_nbr = 1000;

G_mat = [0.7, -0.5; 0.3, 0.9];
R_mat = eye(2);
F_mat = [0.3, 1; 2, -5];

mu_eta = ones(2, 1);
Sigma_eta = eye(2);

Gamma_eta = [0.5, 1.2; -1.8, 2];
nu_eta = ones(2, 1);
Delta_eta = eye(2);
mu_eps = zeros(2, 1);
Sigma_eps = eye(2);

% Simulate the data
state_X = ones(2, 1);
data_Y = zeros(y_nbr, obs_nbr);

eta_vec = csnRandDirect(
    obs_nbr, mu_eta, Sigma_eta, Gamma_eta, nu_eta, Delta_eta
);

eps_vec = mu_eps + chol(Sigma_eps, "lower") * randn(y_nbr, obs_nbr);

for ii = 1:obs_nbr 
    state_X = G_mat * state_X + R_mat * eta_vec(:, ii); 
    data_Y(:, ii) = F_mat * state_X + eps_vec(:, ii);
end


% Function to be given to the optimization procedure
mu_tm1_tm1=zeros(2, 1);
Sigma_tm1_tm1=[10, 0; 0, 10];
Gamma_tm1_tm1=zeros(2, 2);
nu_tm1_tm1=zeros(2, 1);
Delta_tm1_tm1=eye(2);
logcdfmvna_fct="logcdf_ME";
cut_tol=0.1;
skip_lik=false;
skip_loss=true;
loss_fct=nan;
     
neg_log_likeli = @(param) -1 * kalman_csn( ...
    data_Y, ...
    mu_tm1_tm1, ...
    Sigma_tm1_tm1, ...
    Gamma_tm1_tm1, ...
    nu_tm1_tm1, ...
    Delta_tm1_tm1,...
    G_mat, ...
    R_mat, ...
    F_mat, ...
    mu_eta, ...
    Sigma_eta, ...
    [param(1), param(2); -1.8, 2], ...
    nu_eta, ...
    Delta_eta,...
    mu_eps, ...
    Sigma_eps, ...
    logcdfmvna_fct, ...
    cut_tol, ...
    skip_lik, ...
    skip_loss, ...
    loss_fct ...
);


% Minimize the negative log-likelihood to estimate the parameter
res = fminsearch(neg_log_likeli, [-5, 5], optimset('Display', 'iter'));
disp(res)
