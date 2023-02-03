function [log_lik,x_filt,pred,filt] = kalman_csn(Y,...
                                                 mu_tm1_tm1,Sigma_tm1_tm1,Gamma_tm1_tm1,nu_tm1_tm1,Delta_tm1_tm1,...
                                                 G,R,F,...
                                                 mu_eta,Sigma_eta,Gamma_eta,nu_eta,Delta_eta,...
                                                 mu_eps,Sigma_eps,...
                                                 logcdfmvna_fct,cut_tol,skip_lik,skip_loss,loss_fct)
% [log_lik,x_filt,pred,filt] = kalman_csn(DATA,mu_tm1_tm1,Sigma_tm1_tm1,Gamma_tm1_tm1,nu_tm1_tm1,Delta_tm1_tm1,G,R,F,mu_eta,Sigma_eta,Gamma_eta,nu_eta,Delta_eta,mu_eps,Sigma_eps,logcdfmvna_fct,cut_tol,skip_lik,skip_loss,loss_fct)
% -------------------------------------------------------------------------
% Evaluate log-likelihood value of linear state space model with csn distributed innovations and normally distributed noise:
%   x(t) = G*x(t-1) + R*eta(t)  [state transition equation]
%   y(t) = F*x(t)   + eps(t)    [observation equation]
%   eta(t) ~ CSN(mu_eta,Sigma_eta,Gamma_eta,nu_eta,Delta_eta) [innovations, shocks]
%   eps(t) ~ N(mu_eps,Sigma_eps) [noise, measurement error]
% Dimensions:
%   x(t) is (x_nbr by 1) state vector
%   y(t) is (y_nbr by 1) control vector, i.e. observable variables
%   eta(t) is (eta_nbr by 1) vector of innovations
%   eps(t) is (y_nbr by 1) vector of noise (measurement errors)
% -------------------------------------------------------------------------
% INPUTS:
%   - Y               [y_nbr by obs_nbr]             matrix with data
%   - mu_tm1_tm1      [x_nbr by 1]                   initial value of location parameter of CSN distributed states x (does not equal expectation vector unless Gamma_tm1_tm1=0)
%   - Sigma_tm1_tm1   [x_nbr by x_nbr]               initial value of scale parameter of CSN distributed states x (does not equal covariance matrix unless Gamma_tm1_tm1=0)
%   - Gamma_tm1_tm1   [skewx_dim by x_nbr]           initial value of first skewness parameter of CSN distributed states x
%   - nu_tm1_tm1      [skewx_dim by x_nbr]           initial value of second skewness parameter of CSN distributed states x
%   - Delta_tm1_tm1   [skewx_dim by skewx_dim]       initial value of third skewness parameter of CSN distributed states x
%   - G:              [x_nbr by x_nbr]               state transition matrix mapping previous states to current states
%   - R:              [x_nbr by eta_nbr]             state transition matrix mapping current innovations to current states
%   - F:              [y_nbr by x_nbr]               observation equation matrix mapping current states into current observables
%   - mu_eta:         [eta_nbr by 1]                 location parameter of CSN distributed innovations eta (does not equal expectation vector unless Gamma_eta=0)
%   - Sigma_eta:      [eta_nbr by eta_nbr]           scale parameter of CSN distributed innovations eta (does not equal covariance matrix unless Gamma_eta=0)
%   - Gamma_eta:      [skeweta_dim by eta_nbr]       first skewness parameter of CSN distributed innovations eta
%   - nu_eta:         [skeweta_dim by 1]             second skewness parameter of CSN distributed innovations eta
%   - Delta_eta:      [skeweta_dim by skeweta_dim]   third skewness parameter of CSN distributed innovations eta
%   - mu_eps:         [y_nbr by 1]                   location parameter of normally distributed measurement errors eps (equals expectation vector)
%   - Sigma_eps:      [y_nbr by y_nbr]               scale parameter of normally distributed measurement errors eps (equals covariance matrix)
%   - logcdfmvna_fct:    [string]                       name of function with which the log of multivariate normal cdf is calculated, possible values:
%                                                    - 'mvncdf': builtin MATLAB functionality to evaluate approximate CDF, i.e. for bivariate and trivariate distributions, mvncdf uses adaptive quadrature on a transformation of the t density, based on methods developed by Drezner (1994) and Drezner and Wesolowsky (1989) and by Genz (2004). For four or more dimensions, mvncdf uses a quasi-Monte Carlo integration algorithm based on methods developed by Genz (2004) and Genz and Bretz (1999).
%                                                    - 'cdfmvna_ME': evaluate approximate log(CDF) according to Mendell and Elston (1974), implemented by Dietmar Bauer, University Bielefeld
%                                                    - 'cdfmvna_SJ2': evaluate approximate log(CDF) according to Solow (1990) and Joe (1995), implemented by Dietmar Bauer, University Bielefeld
%   - cut_tol         [double]                       correlation threshold to cut redundant skewness dimensions as outlined in the paper, if set to 0 no cutting will be done
%   - skip_lik        [boolean]                      1: skip log-likelihood computations (for doing filtering only)
%   - skip_loss       [boolean]                      1: skip loss function computations (useful for getting uncut filtered states which are then used for smoothing)
%   - loss_fct        [structure]                    Underlying loss function for computing point estimate of filtered states x_t_t that minimizes the expected loss with fields:
%                                                    - type.L0    [boolean] xtilde = mode(x_t_t), i.e. zero-one loss 1*(abs(xtilde-x)<d) for d->0, d is taken from loss_fct.params.d
%                                                    - type.L1    [boolean] xtilde = median(x_t_t), i.e. absolute loss abs(xtilde-x)
%                                                    - type.L2    [boolean] xtilde = E[x_t_t], i.e. squared loss (xtilde-x)^2
%                                                    - type.Lasym [boolean] xtilde = quantile(x_t_t,a/(a+b)), i.e. asymmetric loss function a*abs(xtilde-x) for x>xtilde and b*abs(xtilde-x) for x<=xtilde, , a and b are taken from loss_fct.params.a and loss_fct.params.b
% -------------------------------------------------------------------------
% OUTPUTS:
%   - log_lik:        [scalar]                       value of log likelihood
%   - x_filt:         [structure]                    filtered states according to different loss functions given by loss_fct.type
%   - pred:           [structure]                    csn parameters (mu_t_tm1,Sigma_t_tm1,Gamma_t_tm1,nu_t_tm1,Delta_t_tm1) of predicted states
%   - filt:           [structure]                    csn parameters (mu_t_t,Sigma_t_t,Gamma_t_t,nu_t_t,Delta_t_t) of filtered states
% =========================================================================
% This file is part of the replication files for
% Guljanov, Mutschler, Trede (2022) - Pruned Skewed Kalman Filter and Smoother: With Application to the Yield Curve
%
% Copyright (C) 2022 Gaygysyz Guljanov, Willi Mutschler, Mark Trede
%
% This is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This file is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License <https://www.gnu.org/licenses/>
% for more details.
% =========================================================================
% some settings
kalman_tol = 1e-10; % numerical tolerance for determining the singularity of the covariance matrix of the prediction errors during the Kalman filter (minimum allowed reciprocal of the matrix condition number)
rescale_prediction_error_covariance=false; % rescales the prediction error covariance in the Kalman filter to avoid badly scaled matrix and reduce the probability of a switch to univariate Kalman filters (which are slower). By default no rescaling is done.
Omega_singular = true;
rescale_prediction_error_covariance0=rescale_prediction_error_covariance;

% get dimensions
[y_nbr,x_nbr] = size(F);
obs_nbr = size(Y,2);
%skeweta_nbr = size(Gamma_eta, 1);

% initialize some matrices
mu_eta    = R*mu_eta;
Sigma_eta = R*Sigma_eta*R';
Gamma_eta = Gamma_eta/(R'*R)*R';
Gamma_eta_X_Sigma_eta = Gamma_eta*Sigma_eta;
Delta22_common = Delta_eta + Gamma_eta_X_Sigma_eta*Gamma_eta';
const2pi = -0.5*y_nbr*log(2*pi);

if nargout > 1
    x_filt.L0 = nan(x_nbr,obs_nbr);
    x_filt.L1 = nan(x_nbr,obs_nbr);
    x_filt.L2 = nan(x_nbr,obs_nbr);
    x_filt.Lasym = nan(x_nbr,obs_nbr);    
end
if nargout > 2
    % initialize "pred" structure to save parameters of predicted states
    pred.mu = zeros(x_nbr, obs_nbr);
    pred.Sigma = zeros(x_nbr, x_nbr, obs_nbr);
    pred.Gamma = cell(obs_nbr, 1);
    pred.nu = cell(obs_nbr, 1);
    pred.Delta = cell(obs_nbr, 1);

    % initialize "filt" structure to save parameters of filtered states
    filt.mu = zeros(x_nbr, obs_nbr);
    filt.Sigma = zeros(x_nbr, x_nbr, obs_nbr);
    filt.Gamma = cell(obs_nbr, 1);
    filt.nu = cell(obs_nbr, 1);
    filt.Delta = cell(obs_nbr, 1);
end
log_lik_t = zeros(obs_nbr,1); % vector of likelihood contributions
log_lik = -Inf; % default value of log likelihood

for t=1:obs_nbr
    % auxiliary matrices
    Gamma_tm1_tm1_X_Sigma_tm1_tm1 = Gamma_tm1_tm1*Sigma_tm1_tm1;
    Gamma_tm1_tm1_X_Sigma_tm1_tm1_X_GT = Gamma_tm1_tm1_X_Sigma_tm1_tm1*G';

    % prediction
    mu_t_tm1  = G*mu_tm1_tm1 + mu_eta;
    Sigma_t_tm1 = G*Sigma_tm1_tm1*G' + Sigma_eta;
    Sigma_t_tm1 = 0.5*(Sigma_t_tm1 + Sigma_t_tm1'); % ensure symmetry
    invSigma_t_tm1 = inv(Sigma_t_tm1);
    Gamma_t_tm1 = [Gamma_tm1_tm1_X_Sigma_tm1_tm1_X_GT; Gamma_eta_X_Sigma_eta]*invSigma_t_tm1;
    nu_t_tm1 = [nu_tm1_tm1; nu_eta];
    Delta11_t_tm1 = Delta_tm1_tm1 + Gamma_tm1_tm1_X_Sigma_tm1_tm1*Gamma_tm1_tm1' - Gamma_tm1_tm1_X_Sigma_tm1_tm1_X_GT*invSigma_t_tm1*Gamma_tm1_tm1_X_Sigma_tm1_tm1_X_GT';
    Delta22_t_tm1 = Delta22_common - Gamma_eta_X_Sigma_eta*invSigma_t_tm1*Gamma_eta_X_Sigma_eta';
    Delta12_t_tm1 = -Gamma_tm1_tm1_X_Sigma_tm1_tm1_X_GT*invSigma_t_tm1*Gamma_eta_X_Sigma_eta';
    Delta_t_tm1 = [Delta11_t_tm1 , Delta12_t_tm1; Delta12_t_tm1' , Delta22_t_tm1];
    Delta_t_tm1 = 0.5*(Delta_t_tm1 + Delta_t_tm1'); % ensure symmetry
    y_predicted = F*mu_t_tm1 + mu_eps;
    prediction_error = Y(:,t) - y_predicted;

    % Cutting redundant skewness dimension to speed up filtering
    if cut_tol > 0
        [Sigma_t_tm1, Gamma_t_tm1, nu_t_tm1, Delta_t_tm1] = dim_red4(Sigma_t_tm1,Gamma_t_tm1,nu_t_tm1,Delta_t_tm1,cut_tol);
    end

    % Kalman gains
    Omega = F*Sigma_t_tm1*F' + Sigma_eps;
    Omega = 0.5*(Omega + Omega'); %ensure symmetry
    badly_conditioned_Omega = false;
    if rescale_prediction_error_covariance        
        sig=sqrt(diag(Omega));
        if any(diag(Omega)<kalman_tol) || rcond(Omega./(sig*sig'))<kalman_tol
            badly_conditioned_Omega = true;
            warning('badly_conditioned_Omega')
        end
    else
        if rcond(Omega)<kalman_tol
            sig=sqrt(diag(Omega));
            if any(diag(Omega)<kalman_tol) || rcond(Omega./(sig*sig'))<kalman_tol
                badly_conditioned_Omega = true;
                warning('badly_conditioned_Omega')
            else
                rescale_prediction_error_covariance=1;
                warning('set rescale_prediction_error_covariance to 1')
            end
        end
    end

    if badly_conditioned_Omega
        if ~all(abs(Omega(:))<kalman_tol)
            % Use univariate filter (will remove observations with zero variance prediction error)
            warning('univariate filter not yet for CSN')
            return
        else
            % Pathological case, discard draw.
            warning('discard draw due to badly_conditioned_Omega')
            return
        end
    else
        Omega_singular = false;
        if rescale_prediction_error_covariance
            log_detOmega = log(det(Omega./(sig*sig')))+2*sum(log(sig));
            invOmega = inv(Omega./(sig*sig'))./(sig*sig');
            rescale_prediction_error_covariance=rescale_prediction_error_covariance0;
        else
            log_detOmega = log(det(Omega));
            invOmega = inv(Omega);
        end
        K_Gauss = Sigma_t_tm1*F'*invOmega;
        K_Skewed = Gamma_t_tm1*K_Gauss;
    
        % log-likelihood contributions
        if ~skip_lik
            % The conditional distribution of y(t) given y(t-1) is:
            % (y(t)|y(t-1)) ~ CSN(mu_y,Sigma_y,Gamma_y,nu_y,Delta_y)
            %               = mvncdf(Gamma_y*(y(t)-mu_y),nu_y,Delta_y) / mvncdf(0,nu_y,Delta_y+Gamma_y*Sigma_y*Gamma_y') * mvnpdf(y(t),mu_y,Sigma_y)
            % where:
            %   mu_y    = F*mu_t_tm1 + mu_eps = y_predicted
            %   Sigma_y = F*Sigma_t_tm1*F' + Sigma_eps = Omega
            %   Gamma_y = Gamma_t_tm1*Sigma_t_tm1*F'*inv(F*Sigma_t_tm1*F' + Sigma_eps) = K_Skewed
            %   nu_y    = nu_t_tm1
            %   Delta_y = Delta_t_tm1 + Gamma_t_tm1*Sigma_t_tm1*Gamma_t_tm1'...
            %             - Gamma_t_tm1*Sigma_t_tm1*F'*inv(F*Sigma_t_tm1*F')*F*Sigma_t_tm1*Gamma_t_tm1'...
            %             + (Gamma_t_tm1*Sigma_t_tm1*F'*inv(F*Sigma_t_tm1*F') - Gamma_t_tm1*Sigma_t_tm1*F'*inv(F*Sigma_t_tm1*F' + Sigma_eps))*F*Sigma_t_tm1*Gamma_t_tm1';
            %           = Delta_t_tm1 + (Gamma_t_tm1-K_Skewed*F)*Sigma_t_tm1*Gamma_t_tm1'
            Delta_y = Delta_t_tm1 + (Gamma_t_tm1-K_Skewed*F)*Sigma_t_tm1*Gamma_t_tm1';
            Delta_y = 0.5*(Delta_y + Delta_y'); %ensure symmetry
    
            % evaluate Gaussian cdfs, i.e.
            %  - bottom one: mvncdf(0,nu_y,Delta_y + Gamma_y*Sigma_y*Gamma_y')
            %  - top one: mvncdf(Gamma_y*(y(t)-mu_y),nu_y,Delta_y)
            cdf_bottom_cov = Delta_y + K_Skewed*Omega*K_Skewed';
            cdf_bottom_cov = 0.5*(cdf_bottom_cov + cdf_bottom_cov'); % ensure symmetry
            if strcmp(logcdfmvna_fct,'cdfmvna_ME') || strcmp(logcdfmvna_fct,'cdfmvna_SJ2') || strcmp(logcdfmvna_fct,'logcdf_ME') || strcmp(logcdfmvna_fct,'logcdf_ME_mex')
                % cdfmvna_ME and cdfmvna_SJ2 require zero mean and take correlation matrix instead of covariance matrix as input
                tmp_cov2corr = diag(1./sqrt(diag(cdf_bottom_cov)));
                cdf_bottom_cov = tmp_cov2corr*cdf_bottom_cov*tmp_cov2corr;
                cdf_bottom_cov = 0.5*(cdf_bottom_cov + cdf_bottom_cov'); % ensure symmetry
                if ~isempty(cdf_bottom_cov)
                    log_gaussian_cdf_bottom = feval(logcdfmvna_fct, -tmp_cov2corr*nu_t_tm1, cdf_bottom_cov);
                    if strcmp(logcdfmvna_fct,'cdfmvna_ME') || strcmp(logcdfmvna_fct,'cdfmvna_SJ2')
                        log_gaussian_cdf_bottom = log(log_gaussian_cdf_bottom);
                    end
                else
                    log_gaussian_cdf_bottom = 0;
                end
    
                tmp_cov2corr = diag(1./sqrt(diag(Delta_y)));
                cdf_top_cov = tmp_cov2corr*Delta_y*tmp_cov2corr;
                cdf_top_cov = 0.5*(cdf_top_cov + cdf_top_cov');
                if ~isempty(cdf_top_cov)
                    log_gaussian_cdf_top = feval(logcdfmvna_fct, tmp_cov2corr*(K_Skewed*prediction_error - nu_t_tm1), cdf_top_cov);
                    if strcmp(logcdfmvna_fct,'cdfmvna_ME') || strcmp(logcdfmvna_fct,'cdfmvna_SJ2')
                        log_gaussian_cdf_top = log(log_gaussian_cdf_top);
                    end
                else
                    log_gaussian_cdf_top = 0;
                end
            elseif strcmp(logcdfmvna_fct,'mvncdf')
                log_gaussian_cdf_bottom = log(mvncdf(zeros(size(nu_t_tm1,1),1), nu_t_tm1, cdf_bottom_cov));
                log_gaussian_cdf_top = log(mvncdf(K_Skewed*prediction_error, nu_t_tm1, Delta_y));
            elseif strcmp(logcdfmvna_fct,'qsilatmvnv')
                k = size(nu_t_tm1,1);
                if k > 1
                    log_gaussian_cdf_bottom = log(qsilatmvnv( 1000*k, cdf_bottom_cov, repmat(-Inf,k,1)-nu_t_tm1, zeros(k,1)-nu_t_tm1 ));
                    log_gaussian_cdf_top = log(qsilatmvnv( 1000*k, Delta_y, repmat(-Inf,k,1)-nu_t_tm1, K_Skewed*prediction_error-nu_t_tm1 ));
                else
                    log_gaussian_cdf_bottom = log(normcdf(0, nu_t_tm1, cdf_bottom_cov));
                    log_gaussian_cdf_top = log(normcdf(K_Skewed*prediction_error, nu_t_tm1, Delta_y));
                end
            end
    
            % evaluate Gaussian pdf
            %log_gaussian_pdf = log(mvnpdf(Y(:,t), y_predicted, Omega));
            log_gaussian_pdf = const2pi - 0.5*log_detOmega - 0.5*transpose(prediction_error)*invOmega*prediction_error;
    
            log_lik_t(t) = -log_gaussian_cdf_bottom + log_gaussian_pdf + log_gaussian_cdf_top;
            if isnan(log_lik_t(t))
                log_lik = -Inf;
                x_filt  = nan;
                return
            end
        end

        % filtering
        mu_t_t = mu_t_tm1 + K_Gauss*prediction_error;
        Sigma_t_t = Sigma_t_tm1 - K_Gauss*F*Sigma_t_tm1;
        Gamma_t_t = Gamma_t_tm1;
        nu_t_t = nu_t_tm1 - K_Skewed*prediction_error;
        Delta_t_t = Delta_t_tm1;        
        Sigma_t_t = 0.5*(Sigma_t_t + Sigma_t_t'); % ensure symmetry
        Delta_t_t = 0.5*(Delta_t_t + Delta_t_t'); % ensure symmetry
        if nargout > 1 && ~skip_loss % save the point estimate that minimizes loss of x_t_t
            if loss_fct.type.L0
                % 0-1 loss, i.e. compute mode of CSN distributed x_t_t            
                x_filt.L0(:,t) = csnMode(mu_t_t, Sigma_t_t, Gamma_t_t, nu_t_t, Delta_t_t);
            end
            if loss_fct.type.L1
                % absolue loss, i.e. compute median of CSN distributed x_t_t
                x_filt.L1(:,t) = csnQuantile(0.5, mu_t_t, Sigma_t_t, Gamma_t_t, nu_t_t, Delta_t_t);
            end
            if loss_fct.type.L2
                % squared loss, i.e. compute mean of CSN distributed x_t_t
                if strcmp(logcdfmvna_fct,'logcdf_ME')
                    cdfmvna_fct = 'cdfmvna_ME';
                end
                x_filt.L2(:,t) = csnMean(mu_t_t, Sigma_t_t, Gamma_t_t, nu_t_t, Delta_t_t,cdfmvna_fct);
            end
            if loss_fct.type.Lasym
                % asymmetric loss, i.e. compute a/(a+b) quantile of CSN distributed x_t_t
                x_filt.Lasym(:,t) = csnQuantile(loss_fct.params.a/(loss_fct.params.a+loss_fct.params.b), mu_t_t, Sigma_t_t, Gamma_t_t, nu_t_t, Delta_t_t);
            end
        end

        % assign for next time step
        mu_tm1_tm1 = mu_t_t;
        Sigma_tm1_tm1 = Sigma_t_t;
        Gamma_tm1_tm1 = Gamma_t_t;
        nu_tm1_tm1 = nu_t_t;
        Delta_tm1_tm1 = Delta_t_t;

        if nargout > 2
            % save the parameters of the predicted and filtered csn states
            pred.mu(:,t) = mu_t_tm1;
            pred.Sigma(:,:,t) = Sigma_t_tm1;
            pred.Gamma{t,1} = Gamma_t_tm1;
            pred.nu{t,1} = nu_t_tm1;
            pred.Delta{t,1} = Delta_t_tm1;        
            filt.mu(:,t) = mu_t_t;
            filt.Sigma(:,:,t) = Sigma_t_t;
            filt.Gamma{t,1} = Gamma_t_t;
            filt.nu{t,1} = nu_t_t;
            filt.Delta{t,1} = Delta_t_t;
        end
    end
end

if Omega_singular
    error('The variance of the forecast error remains singular until the end of the sample')
end

% compute log-likelihood by summing individual contributions
log_lik = sum(log_lik_t);

end % main function end