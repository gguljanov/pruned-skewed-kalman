using SpecialFunctions
using LinearAlgebra
using Distributions

# 
# Normal PDF
# 
function phip(z::T) where T<:Real
    # Based on MATLAB codes provided by Dietmar Bauer, Bielefeld University

    p = ℯ^(-z^2 / 2) / √(2π) # Normal pdf

    return p
end


# 
# Normal CDF
# 
function phid(z::T) where T<:Real
    # 
    # taken from Alan Gentz procedures.
    # 
    # Based on MATLAB codes provided by Dietmar Bauer, Bielefeld University

    p = erfc(-z / √2) / 2 # Normal cdf

    return p
end


function logcdf_ME(Zj, Corr_mat)
    # cdf_ME  : Evaluate approximate log(CDF) according to Mendell Elston
    # Zj : A column vector of points where CDF is evaluated, of size (len_cdf)
    #  Corr_mat: Correlation matrix of size (len_cdf) x (len_cdf)
    # 
    # Based on MATLAB codes provided by Dietmar Bauer, Bielefeld University

    cutoff = 6

    Zj[Zj.>cutoff] .= cutoff
    Zj[Zj.<-cutoff] .= -cutoff

    len_cdf = length(Zj) # dimension of CDF.

    cdf_val = phid(Zj[1])
    pdf_val = phip(Zj[1])

    log_res = log(cdf_val) # perform all calcs in logs.

    for _ ∈ 1:(len_cdf-1)
        ajjm1 = pdf_val / cdf_val

        # Update Zj and Rij
        tZ = Zj + ajjm1 * Corr_mat[:, 1]    # update Zj

        R_jj = Corr_mat[:, 1] * Corr_mat[1, :]'
        tRij = Corr_mat - R_jj * (ajjm1 + Zj[1]) * ajjm1    # update Rij

        # Convert Rij (i.e. Covariance matrix) to Correlation matrix
        cov2corr = sqrt.(diag(tRij))

        Zj = tZ ./ cov2corr
        Corr_mat = tRij ./ (cov2corr * cov2corr')

        # Cutoff those dimensions if they are too low to be evaluated
        cutoff = 38

        Zj[Zj.>cutoff] .= cutoff
        Zj[Zj.<-cutoff] .= -cutoff

        # Evaluate jj's probability
        cdf_val = phid(Zj[2])
        pdf_val = phip(Zj[2])

        # Delete unnecessary parts of updated Zj and Corr_mat
        Zj = Zj[2:end]
        Corr_mat = Corr_mat[2:end, :]
        Corr_mat = Corr_mat[:, 2:end]

        # Overall probability
        log_res = log_res + log(cdf_val)
    end

    return log_res
end


function dim_red4(Sigma, Gamma, nu, Delta, cut_tol)
    # Reduces the dimension of csn 
    # according to the correlations of the conditions

    P  = vcat(
        hcat(Sigma, Sigma * Gamma'),
        hcat(Gamma * Sigma, Delta + Gamma * Sigma * Gamma')
    );
    P  = 0.5*(P + P');

    n  = size(Sigma)[1];
    n2 = size(P)[1];

    try
        stdnrd = diagm(1 ./ sqrt.(diag(P)));

        Pcorr  = abs.(stdnrd * P * stdnrd);
    catch
        
        Sigma = 0;
        Gamma = 0;
        nu = 0;
        Delta = 0;
        
        return Sigma, Gamma, nu, Delta
    end

    Pcorr = Pcorr - Matrix(Inf*I, n2, n2)

    logi2 = findmax(Pcorr[n+1:end, 1:n], dims=2)[1];

    logi2 = vec((logi2 .> cut_tol));

    # Cut unnecessary dimensions
    Gamma = Gamma[logi2, :]
    nu = nu[logi2]
    Delta = Delta[logi2, :]
    Delta = Delta[:, logi2]

    return Sigma, Gamma, nu, Delta
end


function (kalman_csn(;
    Y::Matrix{T},
    mu_tm1_tm1::Vector{T},
    Sigma_tm1_tm1::Matrix{T},
    Gamma_tm1_tm1::Matrix{T},
    nu_tm1_tm1::Vector{T},
    Delta_tm1_tm1::Matrix{T},
    G::Matrix{T},
    R::Matrix{T},
    F::Matrix{T},
    mu_eta::Vector{T},
    Sigma_eta::Matrix{T},
    Gamma_eta::Matrix{T},
    nu_eta::Vector{T},
    Delta_eta::Matrix{T},
    mu_eps::Vector{T},
    Sigma_eps::Matrix{T},
    cut_tol::T=0.01,
    eval_lik::Bool=true,
    ret_pred_filt::Bool=false,
    logcdfmvna_fct::Function=logcdf_ME
) where T<:Real)

    """
    -------------------------------------------------------------------------
    Evaluate log-likelihood value of linear state space model with csn distributed innovations and normally distributed noise:
      x[t] = G*x[t-1] + R*eta[t]  [state transition equation]
      y[t] = F*x[t]   + eps[t]    [observation equation]
      eta[t] ~ CSN(
            mu_eta, Sigma_eta, Gamma_eta, nu_eta, Delta_eta
        ) [innovations, shocks]
      eps(t) ~ N(mu_eps,Sigma_eps) [noise, measurement error]
    Dimensions:
      x[t] is (x_nbr by 1) state vector
      y[t] is (y_nbr by 1) control vector, i.e. observable variables
      eta[t] is (eta_nbr by 1) vector of innovations
      eps[t] is (y_nbr by 1) vector of noise (measurement errors)
    -------------------------------------------------------------------------
    INPUTS:
      - Y               
        [y_nbr by obs_nbr]             
        matrix with data
      - mu_tm1_tm1      
        [x_nbr by 1]                   
        initial value of location parameter of CSN distributed states x 
        (does not equal expectation vector unless Gamma_tm1_tm1=0)
      - Sigma_tm1_tm1   
        [x_nbr by x_nbr]               
        initial value of scale parameter of CSN distributed states x 
        (does not equal covariance matrix unless Gamma_tm1_tm1=0)
      - Gamma_tm1_tm1   
        [skewx_dim by x_nbr]           
        initial value of first skewness parameter of CSN distributed states x
      - nu_tm1_tm1      
        [skewx_dim by x_nbr]           
        initial value of second skewness parameter of CSN distributed states x
      - Delta_tm1_tm1   
        [skewx_dim by skewx_dim]       
        initial value of third skewness parameter of CSN distributed states x
      - G:              
        [x_nbr by x_nbr]               
        state transition matrix mapping previous states to current states
      - R:              
        [x_nbr by eta_nbr]             
        state transition matrix mapping current innovations to current states
      - F:              
        [y_nbr by x_nbr]               
        observation equation matrix mapping current states into 
        current observables
      - mu_eta:         
        [eta_nbr by 1]                 
        location parameter of CSN distributed innovations eta 
        (does not equal expectation vector unless Gamma_eta=0)
      - Sigma_eta:      
        [eta_nbr by eta_nbr]           
        scale parameter of CSN distributed innovations eta 
        (does not equal covariance matrix unless Gamma_eta=0)
      - Gamma_eta:      
        [skeweta_dim by eta_nbr]       
        first skewness parameter of CSN distributed innovations eta
      - nu_eta:         
        [skeweta_dim by 1]             
        second skewness parameter of CSN distributed innovations eta
      - Delta_eta:      
        [skeweta_dim by skeweta_dim]   
        third skewness parameter of CSN distributed innovations eta
      - mu_eps:         
        [y_nbr by 1]                   
        location parameter of normally distributed measurement errors eps 
        (equals expectation vector)
      - Sigma_eps:      
        [y_nbr by y_nbr]               
        scale parameter of normally distributed measurement errors eps 
        (equals covariance matrix)
      - logcdfmvna_fct:    
        [Function]                       
        name of function 
        with which the log of multivariate normal cdf is calculated
      - cut_tol         
        [scalar]                       
        correlation threshold to cut redundant skewness dimensions 
        as outlined in the paper, if set to 0 no cutting will be done
      - eval_lik 
        [boolean]                      
        true: carries out the log-likelihood computations 
    -------------------------------------------------------------------------
    OUTPUTS:
      - log_lik:        
        [scalar]                       
        value of log likelihood
      - pred:           
        [dictionary]                    
        csn parameters 
        (mu_t_tm1, Sigma_t_tm1, Gamma_t_tm1, nu_t_tm1, Delta_t_tm1) 
        of predicted states
      - filt:           
        [dictionary]                    
        csn parameters (mu_t_t, Sigma_t_t, Gamma_t_t, nu_t_t, Delta_t_t) 
        of filtered states
    =========================================================================
    This file is part of the replication files for
    Guljanov, Mutschler, Trede (2022) - Pruned Skewed Kalman Filter and Smoother: With Application to the Yield Curve
   
    Copyright (C) 2022 Gaygysyz Guljanov, Willi Mutschler, Mark Trede
   
    This is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
   
    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License <https://www.gnu.org/licenses/>
    for more details.
    =========================================================================
    """
    # Some settings:

    # numerical tolerance for determining the singularity 
    # of the covariance matrix of the prediction errors 
    # during the Kalman filter 
    # (minimum allowed reciprocal of the matrix condition number)
    kalman_tol = 1e-10

    # rescales the prediction error covariance in the Kalman filter 
    # to avoid badly scaled matrix and reduce the probability of a switch 
    # to univariate Kalman filters (which are slower). 
    # By default no rescaling is done.
    rescale_prediction_error_covariance = false
    Omega_singular = true
    rescale_prediction_error_covariance0 = rescale_prediction_error_covariance

    # get dimensions
    y_nbr, x_nbr = size(F)
    obs_nbr = size(Y, 2)
    #skeweta_nbr = size(Gamma_eta, 1);

    # initialize some matrices
    mu_eta = R * mu_eta
    Sigma_eta = R * Sigma_eta * R'
    Gamma_eta = Gamma_eta / (R' * R) * R'
    Gamma_eta_X_Sigma_eta = Gamma_eta * Sigma_eta
    Delta22_common = Delta_eta + Gamma_eta_X_Sigma_eta * Gamma_eta'
    const2pi = -0.5 * y_nbr * log(2 * pi)

    pred = NaN
    filt = NaN

    if ret_pred_filt
        # initialize "pred" dictionary to save parameters of predicted states
        pred = Dict([("mu", Array{T,2}(undef, x_nbr, obs_nbr)),
            ("Sigma", Array{T,3}(undef, x_nbr, x_nbr, obs_nbr)),
            ("Gamma", Array{Any,1}(undef, obs_nbr)),
            ("nu", Array{Any,1}(undef, obs_nbr)),
            ("Delta", Array{Any,1}(undef, obs_nbr))])

        # initialize "filt" dictionary to save parameters of filtered states
        filt = Dict([("mu", Array{T,2}(undef, x_nbr, obs_nbr)),
            ("Sigma", Array{T,3}(undef, x_nbr, x_nbr, obs_nbr)),
            ("Gamma", Array{Any,1}(undef, obs_nbr)),
            ("nu", Array{Any,1}(undef, obs_nbr)),
            ("Delta", Array{Any,1}(undef, obs_nbr))])
    end

    log_lik_t = zeros(obs_nbr, 1) # vector of likelihood contributions
    log_lik = -Inf # default value of log likelihood

    for t ∈ 1:obs_nbr
        # Auxiliary matrices
        Gamma_tm1_tm1_X_Sigma_tm1_tm1 = Gamma_tm1_tm1 * Sigma_tm1_tm1
        Gamma_tm1_tm1_X_Sigma_tm1_tm1_X_GT = Gamma_tm1_tm1_X_Sigma_tm1_tm1 * G'

        # Prediction step
        mu_t_tm1 = G * mu_tm1_tm1 + mu_eta

        Sigma_t_tm1 = G * Sigma_tm1_tm1 * G' + Sigma_eta
        Sigma_t_tm1 = 0.5 * (Sigma_t_tm1 + Sigma_t_tm1') # ensure symmetry

        invSigma_t_tm1 = inv(Sigma_t_tm1)

        Gamma_t_tm1 = (
            vcat(
                Gamma_tm1_tm1_X_Sigma_tm1_tm1_X_GT,
                Gamma_eta_X_Sigma_eta
            )
            *
            invSigma_t_tm1
        )

        nu_t_tm1 = vcat(nu_tm1_tm1, nu_eta)

        Delta11_t_tm1 = (
            Delta_tm1_tm1
            +
            Gamma_tm1_tm1_X_Sigma_tm1_tm1
            *
            Gamma_tm1_tm1'
            -
            Gamma_tm1_tm1_X_Sigma_tm1_tm1_X_GT
            * invSigma_t_tm1
            * Gamma_tm1_tm1_X_Sigma_tm1_tm1_X_GT'
        )

        Delta22_t_tm1 = (
            Delta22_common
            -
            Gamma_eta_X_Sigma_eta
            * invSigma_t_tm1
            * Gamma_eta_X_Sigma_eta'
        )

        Delta12_t_tm1 = (
            -Gamma_tm1_tm1_X_Sigma_tm1_tm1_X_GT
            * invSigma_t_tm1
            * Gamma_eta_X_Sigma_eta'
        )

        Delta_t_tm1 = vcat(
            hcat(Delta11_t_tm1, Delta12_t_tm1),
            hcat(Delta12_t_tm1', Delta22_t_tm1)
        )

        Delta_t_tm1 = 0.5 * (Delta_t_tm1 + Delta_t_tm1') # ensure symmetry

        y_predicted = F * mu_t_tm1 + mu_eps

        prediction_error = Y[:, t] - y_predicted

        # Cutting redundant skewness dimension to speed up filtering
        if cut_tol > 0
            Sigma_t_tm1, Gamma_t_tm1, nu_t_tm1, Delta_t_tm1 = dim_red4(
                Sigma_t_tm1, Gamma_t_tm1, nu_t_tm1, Delta_t_tm1, cut_tol
            )
        end

        # Kalman gains
        Omega = F * Sigma_t_tm1 * F' + Sigma_eps
        Omega = 0.5 * (Omega + Omega') #ensure symmetry

        badly_conditioned_Omega = false

        if rescale_prediction_error_covariance
            sig = sqrt.(diag(Omega))

            if (
                any(diag(Omega) < kalman_tol)
                ||
                1 / cond((Omega ./ (sig * sig')), 1) < kalman_tol
            )

                badly_conditioned_Omega = true

                @warn "badly_conditioned_Omega"
            end
        else
            if 1/cond(Omega, 1) < kalman_tol
                sig = sqrt.(diag(Omega))

                if (
                    any(diag(Omega) < kalman_tol)
                    ||
                    cond((Omega ./ (sig * sig')), 1) < kalman_tol
                )
                    badly_conditioned_Omega = true

                    @warn "badly_conditioned_Omega"
                else
                    rescale_prediction_error_covariance = 1

                    @warn "set rescale_prediction_error_covariance to 1"
                end
            end
        end

        if badly_conditioned_Omega
            if !all(abs.(Omega) .< kalman_tol)
                # Use univariate filter 
                # (will remove observations with zero variance prediction error)
                @warn "univariate filter not yet for CSN"
            else
                # Pathological case, discard draw.
                @warn "discard draw due to badly_conditioned_Omega"
            end

            return log_lik, pred, filt
        end

        Omega_singular = false

        if rescale_prediction_error_covariance
            log_detOmega = log(det(Omega ./ (sig * sig'))) + 2 * sum(log(sig))

            invOmega = inv(Omega ./ (sig * sig')) ./ (sig * sig')

            rescale_prediction_error_covariance = rescale_prediction_error_covariance0
        else
            log_detOmega = log(det(Omega))

            invOmega = inv(Omega)
        end

        K_Gauss = Sigma_t_tm1 * F' * invOmega
        K_Skewed = Gamma_t_tm1 * K_Gauss

        # log-likelihood contributions
        if eval_lik
            # The conditional distribution of y[t] given y[t-1] is:
            # (y[t]|y[t-1]) ~ CSN(mu_y, Sigma_y, Gamma_y, nu_y, Delta_y)
            #               = mvncdf(Gamma_y*(y[t]-mu_y), nu_y, Delta_y) 
            #                 / mvncdf(0, 
            #                          nu_y, 
            #                          Delta_y + Gamma_y*Sigma_y*Gamma_y') 
            #                 * mvnpdf(y[t],mu_y,Sigma_y)
            # where:
            #   mu_y    = F*mu_t_tm1 + mu_eps = y_predicted
            #   Sigma_y = F*Sigma_t_tm1*F' + Sigma_eps = Omega
            #   Gamma_y = Gamma_t_tm1*Sigma_t_tm1*F'*
            #             inv(F*Sigma_t_tm1*F' + Sigma_eps) 
            #           = K_Skewed
            #   nu_y    = nu_t_tm1
            #   Delta_y = Delta_t_tm1 + Gamma_t_tm1*Sigma_t_tm1*Gamma_t_tm1'...
            #             - Gamma_t_tm1*Sigma_t_tm1*F'*inv(F*Sigma_t_tm1*F')
            #               *F*Sigma_t_tm1*Gamma_t_tm1'...
            #             + (Gamma_t_tm1*Sigma_t_tm1*F'*inv(F*Sigma_t_tm1*F') 
            #             - Gamma_t_tm1*Sigma_t_tm1*F'
            #               *inv(F*Sigma_t_tm1*F' + Sigma_eps))
            #               *F*Sigma_t_tm1*Gamma_t_tm1';
            #           = Delta_t_tm1 + (Gamma_t_tm1-K_Skewed*F)
            #             *Sigma_t_tm1*Gamma_t_tm1'
            # Here,
            # - mvncdf() and mvnpdf() stand for Matlab's built in functions 
            #   to evaluate multivariate normal cdf and multivairate normal pdf,
            #   i.e. normal cdf and normal pdf

            Delta_y = (
                Delta_t_tm1 +
                (Gamma_t_tm1 - K_Skewed * F) * Sigma_t_tm1 * Gamma_t_tm1'
            )
            Delta_y = 0.5 * (Delta_y + Delta_y') #ensure symmetry

            # evaluate Gaussian cdfs, i.e.
            #  - bottom one: mvncdf(0, nu_y, Delta_y + Gamma_y*Sigma_y*Gamma_y')
            #  - top one: mvncdf(Gamma_y*(y[t]-mu_y), nu_y, Delta_y)
            cdf_bottom_cov = Delta_y + K_Skewed * Omega * K_Skewed'
            cdf_bottom_cov = 0.5 * (cdf_bottom_cov + cdf_bottom_cov')

            if logcdfmvna_fct != logcdf_ME
                @error "Only logcdf_ME has been implemented for now"
            end

            # Evaluate the bottom cdf
            tmp_cov2corr = diagm(1 ./ sqrt.(diag(cdf_bottom_cov)))

            cdf_bottom_cov = tmp_cov2corr * cdf_bottom_cov * tmp_cov2corr
            cdf_bottom_cov = 0.5 * (cdf_bottom_cov + cdf_bottom_cov')

            if !isempty(cdf_bottom_cov)
                log_gaussian_cdf_bottom = logcdfmvna_fct(
                    -tmp_cov2corr * nu_t_tm1, cdf_bottom_cov
                )
            else
                log_gaussian_cdf_bottom = 0
            end

            # Evaluate the top cdf
            tmp_cov2corr = diagm(1 ./ sqrt.(diag(Delta_y)))

            cdf_top_cov = tmp_cov2corr * Delta_y * tmp_cov2corr
            cdf_top_cov = 0.5 * (cdf_top_cov + cdf_top_cov')

            if !isempty(cdf_top_cov)
                log_gaussian_cdf_top = logcdfmvna_fct(
                    tmp_cov2corr * (K_Skewed * prediction_error - nu_t_tm1),
                    cdf_top_cov
                )
            else
                log_gaussian_cdf_top = 0
            end

            # evaluate Gaussian pdf
            #log_gaussian_pdf = log(mvnpdf(Y[:,t], y_predicted, Omega));
            log_gaussian_pdf = (
                const2pi
                -
                0.5 * log_detOmega
                -
                0.5 * transpose(prediction_error) * invOmega * prediction_error
            )

            log_lik_t[t] = (
                -log_gaussian_cdf_bottom
                + log_gaussian_pdf
                + log_gaussian_cdf_top)

            if isnan(log_lik_t[t])
                log_lik = -Inf
                x_filt = nan
                return log_lik, pred, filt
            end
        end

        # Filtering step
        mu_t_t = mu_t_tm1 + K_Gauss * prediction_error

        Sigma_t_t = Sigma_t_tm1 - K_Gauss * F * Sigma_t_tm1

        Gamma_t_t = Gamma_t_tm1

        nu_t_t = nu_t_tm1 - K_Skewed * prediction_error

        Delta_t_t = Delta_t_tm1

        Sigma_t_t = 0.5 * (Sigma_t_t + Sigma_t_t') # ensure symmetry
        Delta_t_t = 0.5 * (Delta_t_t + Delta_t_t') # ensure symmetry

        # assign for next time step
        mu_tm1_tm1 = mu_t_t
        Sigma_tm1_tm1 = Sigma_t_t
        Gamma_tm1_tm1 = Gamma_t_t
        nu_tm1_tm1 = nu_t_t
        Delta_tm1_tm1 = Delta_t_t

        if ret_pred_filt
            # save the parameters of the predicted and filtered csn states
            pred["mu"][:, t] = mu_t_tm1
            pred["Sigma"][:, :, t] = Sigma_t_tm1
            pred["Gamma"][t] = Gamma_t_tm1
            pred["nu"][t] = nu_t_tm1
            pred["Delta"][t] = Delta_t_tm1

            filt["mu"][:, t] = mu_t_t
            filt["Sigma"][:, :, t] = Sigma_t_t
            filt["Gamma"][t] = Gamma_t_t
            filt["nu"][t] = nu_t_t
            filt["Delta"][t] = Delta_t_t
        end
    end

    if Omega_singular
        @error "The variance of the forecast error remains singular until the end of the sample"
    end

    # compute log-likelihood by summing individual contributions
    log_lik = sum(log_lik_t)

    return log_lik, pred, filt
end # main function end
