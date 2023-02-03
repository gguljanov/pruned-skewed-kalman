# The dependencies
import warnings
import os
import numpy as np
import scipy
from scipy import special
from typing import Callable


# Normal pdf -- univariate
def phip(z_val: float) -> float:

    # Based on MATLAB codes provided by Dietmar Bauer, Bielefeld University

    f_z = np.exp(-(z_val**2) / 2) / np.sqrt(2 * np.pi)

    return f_z


# Normal cdf -- univariate
def phid(z_val: float) -> float:

    #
    # taken from Alan Gentz procedures.
    #
    # Based on MATLAB codes provided by Dietmar Bauer, Bielefeld University

    f_z = special.erfc(-z_val / np.sqrt(2)) / 2  # Normal cdf

    return f_z


# Mendel-Elston method for Normal cdf approximation
def logcdf_ME(
        Zj: np.ndarray[float],
        Corr_mat: np.ndarray[float]
) -> float:

    # cdf_ME  : Evaluate approximate log(CDF) according to Mendell Elston
    # Zj      : A column vector (a numpy array of dimension two)
    # of points where CDF is evaluated,
    #           of size (len_cdf)
    # Corr_mat: Correlation matrix (a numpy array of dimension two)
    # of size (len_cdf) x (len_cdf)
    #
    # Based on MATLAB codes provided by Dietmar Bauer, Bielefeld University

    cutoff = 6

    Zj[Zj > cutoff, None] = cutoff  # Keep as col vector with None
    Zj[Zj < -cutoff, None] = -cutoff

    len_cdf = len(Zj)  # dimension of CDF.

    cdf_val = phid(Zj[0, 0])
    pdf_val = phip(Zj[0, 0])

    log_res = np.log(cdf_val)  # perform all calcs in logs.

    if len_cdf < 2:
        return log_res

    for jj in range(len_cdf - 1):

        ajjm1 = pdf_val / cdf_val

        # Update Zj
        tZ = Zj + ajjm1 * Corr_mat[:, 0, None]

        # Update Rij
        R_jj = Corr_mat[:, 0, None] @ Corr_mat[None, 0, :]
        tRij = Corr_mat - R_jj * (ajjm1 + Zj[0, 0]) * ajjm1

        # Convert Rij (i.e. Covariance matrix) to Correlation matrix
        cov2corr = np.sqrt(np.diag(tRij).reshape(-1, 1))

        Zj = tZ / cov2corr
        Corr_mat = tRij / (cov2corr @ cov2corr.T)

        # Cutoff those dimensions if they are too low to be evaluated
        cutoff = 38

        Zj[Zj > cutoff, None] = cutoff
        Zj[Zj < -cutoff, None] = -cutoff

        # Evaluate jj.Ts probability
        cdf_val = phid(Zj[1, 0])
        pdf_val = phip(Zj[1, 0])

        # Delete unnecessary parts of updated Zj and Corr_mat
        Zj = np.delete(Zj, 0, 0)
        Corr_mat = np.delete(Corr_mat, 0, 0)
        Corr_mat = np.delete(Corr_mat, 0, 1)

        # Overall probability
        log_res = log_res + np.log(cdf_val)

    return log_res


def dim_red4(
        Sigma: np.ndarray[float], 
        Gamma: np.ndarray[float], 
        nu: np.ndarray[float], 
        Delta: np.ndarray[float], 
        cut_tol: float
) -> tuple[
    np.ndarray[float], 
    np.ndarray[float], 
    np.ndarray[float], 
    np.ndarray[float]
]:

    # Reduces the dimension of csn
    # according to the correlations of the conditions

    n = Sigma.shape[0]
    P = np.block(
        [[Sigma, Sigma @ Gamma.T], 
         [Gamma @ Sigma, Delta + Gamma @ Sigma @ Gamma.T]]
    )
    P = 0.5 * (P + P.T)
    n2 = P.shape[0]


    try:

        stdnrd = np.diag(1 / np.sqrt(np.diag(P)))
        Pcorr = abs(stdnrd @ P @ stdnrd)

    except:

        Sigma = np.array([[0]])
        Gamma = np.array([[0]])
        nu = np.array([[0]])
        Delta = np.array([[0]])

        return Sigma, Gamma, nu, Delta


    Pcorr = Pcorr - np.diag(np.repeat(np.inf, n2))
    logi2 = np.amax(Pcorr[n:, 0:n], 1)

    logi2 = logi2 < cut_tol

    # Cut unnecessary dimensions
    Gamma = np.delete(Gamma, logi2, 0)
    nu = np.delete(nu, logi2, 0)
    Delta = np.delete(Delta, logi2, 0)
    Delta = np.delete(Delta, logi2, 1)

    return Sigma, Gamma, nu, Delta


def kalman_csn(
    Y: np.ndarray[float],
    mu_tm1_tm1: np.ndarray[float],
    Sigma_tm1_tm1: np.ndarray[float],
    Gamma_tm1_tm1: np.ndarray[float],
    nu_tm1_tm1: np.ndarray[float],
    Delta_tm1_tm1: np.ndarray[float],
    G: np.ndarray[float],
    R: np.ndarray[float],
    F: np.ndarray[float],
    mu_eta: np.ndarray[float],
    Sigma_eta: np.ndarray[float],
    Gamma_eta: np.ndarray[float],
    nu_eta: np.ndarray[float],
    Delta_eta: np.ndarray[float],
    mu_eps: np.ndarray[float],
    Sigma_eps: np.ndarray[float],
    cut_tol: float = 1e-2,
    eval_lik: bool = True,
    ret_pred_filt: bool = False,
    logcdfmvna_fct: Callable = logcdf_ME,
) -> tuple[float, dict, dict]:

    """
    -------------------------------------------------------------------------
    Evaluates log-likelihood value of linear state space model
    with csn distributed innovations and normally distributed noise:

      x[t] = G*x[t-1] + R*eta[t]  [state transition equation]
      y[t] = F*x[t]   + eps[t]    [observation equation]
      eta[t] ~ CSN(mu_eta, Sigma_eta, Gamma_eta, nu_eta, Delta_eta)
                                   [innovations, shocks]
      eps[t] ~ N(mu_eps,Sigma_eps) [noise, measurement error]


    Dimensions:
      x[t] is (x_nbr by 1) state vector
      y[t] is (y_nbr by 1) control vector, i.e. observable variables
      eta[t] is (eta_nbr by 1) vector of innovations
      eps[t] is (y_nbr by 1) vector of noise (measurement errors)


    -------------------------------------------------------------------------
    INPUTS:
      - Y:
        [y_nbr by obs_nbr] 
        matrix with data

      - mu_tm1_tm1:
        [x_nbr by 1] 
        initial value of location parameter 
        of CSN distributed states x (does not equal expectation vector 
        unless Gamma_tm1_tm1=0)

      - Sigma_tm1_tm1 
        [x_nbr by x_nbr] 
        initial value of scale parameter
        of CSN distributed states x (does not equal covariance matrix 
        unless Gamma_tm1_tm1=0)

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
        observation equation matrix mapping current states into current
        observables

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
        [function name]                       
        name of function with which the log of multivariate normal cdf 
        is calculated

      - cut_tol         
        [double]                       
        correlation threshold to cut redundant skewness dimensions as outlined 
        in the paper, if set to 0 no cutting will be done

      - eval_lik        
        [boolean]                      
        True: Carries out log-likelihood computations


    -------------------------------------------------------------------------
    OUTPUTS:
      - log_lik:        
        [scalar]                       
        value of log likelihood

      - pred:           
        [structure]                    
        csn parameters (mu_t_tm1,Sigma_t_tm1,Gamma_t_tm1,nu_t_tm1,Delta_t_tm1) 
        of predicted states

      - filt:           
        [structure]                    
        csn parameters (mu_t_t,Sigma_t_t,Gamma_t_t,nu_t_t,Delta_t_t) 
        of filtered states

    =========================================================================
    This file is part of the replication files for
    Guljanov, Mutschler, Trede (2022) - Pruned Skewed Kalman Filter 
    and Smoother: With Application to the Yield Curve

    Copyright (C) 2022 Gaygysyz Guljanov, Willi Mutschler, Mark Trede

    This is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License <https://www.gnu.org/licenses/>
    for more details.
    """

    # Some settings:

    # 'kalman_tol': numerical tolerance for determining the singularity
    # of the covariance matrix of the prediction errors
    # during the Kalman filter
    # (minimum allowed reciprocal of the matrix condition number)
    kalman_tol = 1e-10

    # 'rescale_prediction_error_covariance': rescales the prediction error
    # covariance in the Kalman filter to avoid badly scaled matrix
    # and reduce the probability of a switch to univariate Kalman filters
    # (which are slower). By default no rescaling is done.
    rescale_prediction_error_covariance = False
    Omega_singular = True
    rescale_prediction_error_covariance0 = rescale_prediction_error_covariance

    # Get dimensions
    y_nbr, x_nbr = F.shape
    obs_nbr = Y.shape[1]
    # skeweta_nbr = Gamma_eta.shape[0]

    # Initialize some matrices
    mu_eta = R @ mu_eta
    Sigma_eta = R @ Sigma_eta @ R.T
    Gamma_eta = Gamma_eta @ np.linalg.solve((R.T @ R), R.T)

    Gamma_eta_X_Sigma_eta = Gamma_eta @ Sigma_eta
    Delta22_common = Delta_eta + Gamma_eta_X_Sigma_eta @ Gamma_eta.T

    const2pi = -0.5 * y_nbr * np.log(2 * np.pi)

    # Initialize dictionaries to save parameters of predicted/filtered states
    pred = {}
    filt = {}

    if ret_pred_filt:
        pred["mu"] = []
        pred["Sigma"] = []
        pred["Gamma"] = []
        pred["nu"] = []
        pred["Delta"] = []

        filt["mu"] = []
        filt["Sigma"] = []
        filt["Gamma"] = []
        filt["nu"] = []
        filt["Delta"] = []

    log_lik_t = np.zeros((obs_nbr, 1))  # vector of likelihood contributions
    log_lik = -np.inf  # default value of log likelihood

    for t in range(obs_nbr):

        # Auxiliary matrices
        Gamma_tm1_tm1_X_Sigma_tm1_tm1 = Gamma_tm1_tm1 @ Sigma_tm1_tm1
        Gamma_tm1_tm1_X_Sigma_tm1_tm1_X_GT = Gamma_tm1_tm1_X_Sigma_tm1_tm1 @ G.T

        # Prediction step
        mu_t_tm1 = G @ mu_tm1_tm1 + mu_eta

        Sigma_t_tm1 = G @ Sigma_tm1_tm1 @ G.T + Sigma_eta
        Sigma_t_tm1 = 0.5 * (Sigma_t_tm1 + Sigma_t_tm1.T)  # ensure symmetry

        invSigma_t_tm1 = np.linalg.inv(Sigma_t_tm1)

        Gamma_t_tm1 = (
            np.block([[Gamma_tm1_tm1_X_Sigma_tm1_tm1_X_GT], 
                      [Gamma_eta_X_Sigma_eta]])
            @ invSigma_t_tm1
        )

        nu_t_tm1 = np.block([[nu_tm1_tm1], [nu_eta]])

        Delta11_t_tm1 = (
            Delta_tm1_tm1
            + Gamma_tm1_tm1_X_Sigma_tm1_tm1 @ Gamma_tm1_tm1.T
            - Gamma_tm1_tm1_X_Sigma_tm1_tm1_X_GT
            @ invSigma_t_tm1
            @ Gamma_tm1_tm1_X_Sigma_tm1_tm1_X_GT.T
        )
        Delta22_t_tm1 = (
            Delta22_common
            - Gamma_eta_X_Sigma_eta @ invSigma_t_tm1 @ Gamma_eta_X_Sigma_eta.T
        )
        Delta12_t_tm1 = (
            -Gamma_tm1_tm1_X_Sigma_tm1_tm1_X_GT
            @ invSigma_t_tm1
            @ Gamma_eta_X_Sigma_eta.T
        )

        Delta_t_tm1 = np.block(
            [[Delta11_t_tm1, Delta12_t_tm1], [Delta12_t_tm1.T, Delta22_t_tm1]]
        )

        Delta_t_tm1 = 0.5 * (Delta_t_tm1 + Delta_t_tm1.T)  # ensure symmetry

        y_predicted = F @ mu_t_tm1 + mu_eps

        prediction_error = Y[:, t, None] - y_predicted

        # Cutting redundant skewness dimension to speed up filtering
        if cut_tol > 0:

            Sigma_t_tm1, Gamma_t_tm1, nu_t_tm1, Delta_t_tm1 = dim_red4(
                Sigma_t_tm1, Gamma_t_tm1, nu_t_tm1, Delta_t_tm1, cut_tol
            )

        # Kalman gains
        Omega = F @ Sigma_t_tm1 @ F.T + Sigma_eps
        Omega = 0.5 * (Omega + Omega.T)  # ensure symmetry

        badly_conditioned_Omega = False

        sig = np.sqrt(np.diag(Omega)).reshape(-1, 1)

        if rescale_prediction_error_covariance:

            if (
                np.any(np.diag(Omega) < kalman_tol)
                or 1 / np.linalg.cond(Omega / (sig @ sig.T), p=1) < kalman_tol
            ):

                badly_conditioned_Omega = True

                warnings.warn("badly_conditioned_Omega")

        else:

            if 1 / np.linalg.cond(Omega, p=1) < kalman_tol:

                if (
                    np.any(np.diag(Omega) < kalman_tol)
                    or 
                    1 / np.linalg.cond(Omega / (sig @ sig.T), p=1) < kalman_tol
                ):

                    badly_conditioned_Omega = True

                    warnings.warn("badly_conditioned_Omega")

                else:

                    rescale_prediction_error_covariance = 1
                    warnings.warn("set rescale_prediction_error_covariance" 
                                  + "to 1")

        if badly_conditioned_Omega:
            if not np.all(np.abs(Omega.reshape(-1, 1)) < kalman_tol):
                # Use univariate filter
                # (will remove observations with zero variance prediction error)
                warnings.warn("univariate filter not yet for CSN")
                return log_lik, pred, filt
            else:
                # Pathological case, discard draw
                warnings.warn("discard draw due to badly_conditioned_Omega")
                return log_lik, pred, filt

        Omega_singular = False

        if rescale_prediction_error_covariance:

            log_detOmega = np.log(np.linalg.det(Omega / (sig @ sig.T))) + 2 * np.sum(
                np.log(sig)
            )

            invOmega = np.linalg.inv(Omega / (sig @ sig.T)) / (sig @ sig.T)

            rescale_prediction_error_covariance = rescale_prediction_error_covariance0

        else:

            log_detOmega = np.log(np.linalg.det(Omega))

            invOmega = np.linalg.inv(Omega)

        K_Gauss = Sigma_t_tm1 @ F.T @ invOmega
        K_Skewed = Gamma_t_tm1 @ K_Gauss

        # log-likelihood contributions
        if eval_lik:

            # The conditional distribution of y[t] given y[t-1] is:
            # (y[t] | y[t-1]) ~ CSN(mu_y, Sigma_y, Gamma_y, nu_y, Delta_y)
            # = (
            #    mvncdf(Gamma_y * (y[t] - mu_y), nu_y, Delta_y)
            #    / mvncdf(0, nu_y, Delta_y + Gamma_y * Sigma_y * Gamma_y.T)
            #    * mvnpdf(y[t], mu_y, Sigma_y)
            # )
            #
            # where:
            #  mu_y    = F @ mu_t_tm1 + mu_eps = y_predicted
            #  Sigma_y = F @ Sigma_t_tm1 @ F.T + Sigma_eps = Omega
            #  Gamma_y = Gamma_t_tm1 @ Sigma_t_tm1 @ F.T
            #            @ inv(F @ Sigma_t_tm1 @ F.T + Sigma_eps) = K_Skewed
            #  nu_y    = nu_t_tm1
            #  Delta_y = Delta_t_tm1
            #            + Gamma_t_tm1 @ Sigma_t_tm1 @ Gamma_t_tm1.T
            #            - Gamma_t_tm1 @ Sigma_t_tm1 @ F.T
            #              @ inv(F @ Sigma_t_tm1 @ F.T)
            #              @ F @ Sigma_t_tm1 @ Gamma_t_tm1.T
            #            + (Gamma_t_tm1 @ Sigma_t_tm1 @ F.T
            #                 @ inv(F*Sigma_t_tm1*F.T)
            #               - Gamma_t_tm1 @ Sigma_t_tm1 @ F.T
            #                 @ inv(F @ Sigma_t_tm1 @ F.T + Sigma_eps))
            #              @ F @ Sigma_t_tm1 @ Gamma_t_tm1.T
            #          = Delta_t_tm1 + (Gamma_t_tm1 - K_Skewed @ F)
            #            @ Sigma_t_tm1 @ Gamma_t_tm1.T

            Delta_y = (
                Delta_t_tm1 
                + (Gamma_t_tm1 - K_Skewed @ F) @ Sigma_t_tm1 @ Gamma_t_tm1.T
            )
            Delta_y = 0.5 * (Delta_y + Delta_y.T)  # ensure symmetry

            # evaluate Gaussian cdfs, i.e.
            #  - bottom one:
            #    mvncdf(0, nu_y, Delta_y + Gamma_y @ Sigma_y @ Gamma_y.T)
            #  - top one:
            #    mvncdf(Gamma_y @ (y[t] - mu_y), nu_y, Delta_y)
            cdf_bottom_cov = Delta_y + K_Skewed @ Omega @ K_Skewed.T
            cdf_bottom_cov = 0.5 * (cdf_bottom_cov + cdf_bottom_cov.T)

            if not logcdfmvna_fct == logcdf_ME:
                raise Exception(
                    "Only logcdf_ME has been implemented" 
                    + " for normal cdf evaluation"
                )

            # Evaluate bottom cdf
            tmp_cov2corr = np.diag(1 / np.sqrt(np.diag(cdf_bottom_cov)))

            cdf_bottom_cov = tmp_cov2corr @ cdf_bottom_cov @ tmp_cov2corr
            cdf_bottom_cov = 0.5 * (cdf_bottom_cov + cdf_bottom_cov.T)

            if cdf_bottom_cov.size > 0:

                log_gaussian_cdf_bottom = logcdfmvna_fct(
                    -tmp_cov2corr @ nu_t_tm1, cdf_bottom_cov
                )

            else:

                log_gaussian_cdf_bottom = 0

            # Evaluate top cdf
            tmp_cov2corr = np.diag(1 / np.sqrt(np.diag(Delta_y)))

            cdf_top_cov = tmp_cov2corr @ Delta_y @ tmp_cov2corr
            cdf_top_cov = 0.5 * (cdf_top_cov + cdf_top_cov.T)

            if cdf_top_cov.size > 0:

                log_gaussian_cdf_top = logcdfmvna_fct(
                    tmp_cov2corr @ (K_Skewed @ prediction_error - nu_t_tm1), 
                    cdf_top_cov
                )

            else:

                log_gaussian_cdf_top = 0

            # evaluate Gaussian pdf
            # log_gaussian_pdf = log(mvnpdf(Y[:, t], y_predicted, Omega))
            log_gaussian_pdf = (
                const2pi
                - 0.5 * log_detOmega
                - 0.5 * prediction_error.T @ invOmega @ prediction_error
            )

            log_lik_t[t] = (
                -log_gaussian_cdf_bottom 
                + log_gaussian_pdf 
                + log_gaussian_cdf_top
            )

            if np.isnan(log_lik_t[t]):
                raise Exception(
                    "Likelihood contribution is NaN "
                    + "at iteration = "
                    + "{iteration}!".format(iteration=t + 1)
                )

        # Filtering step
        mu_t_t = mu_t_tm1 + K_Gauss @ prediction_error

        Sigma_t_t = Sigma_t_tm1 - K_Gauss @ F @ Sigma_t_tm1

        Gamma_t_t = Gamma_t_tm1

        nu_t_t = nu_t_tm1 - K_Skewed @ prediction_error

        Delta_t_t = Delta_t_tm1

        Sigma_t_t = 0.5 * (Sigma_t_t + Sigma_t_t.T)  # ensure symmetry
        Delta_t_t = 0.5 * (Delta_t_t + Delta_t_t.T)  # ensure symmetry

        # assign for next time step
        mu_tm1_tm1 = mu_t_t
        Sigma_tm1_tm1 = Sigma_t_t
        Gamma_tm1_tm1 = Gamma_t_t
        nu_tm1_tm1 = nu_t_t
        Delta_tm1_tm1 = Delta_t_t

        if ret_pred_filt:
            # save the parameters of the predicted and filtered csn states
            pred["mu"].append(mu_t_tm1)
            pred["Sigma"].append(Sigma_t_tm1)
            pred["Gamma"].append(Gamma_t_tm1)
            pred["nu"].append(nu_t_tm1)
            pred["Delta"].append(Delta_t_tm1)

            filt["mu"].append(mu_t_t)
            filt["Sigma"].append(Sigma_t_t)
            filt["Gamma"].append(Gamma_t_t)
            filt["nu"].append(nu_t_t)
            filt["Delta"].append(Delta_t_t)

    if Omega_singular:
        raise Exception(
            "The variance of the forecast error "
            + "remains singular until the end of the sample"
        )

    # Compute log-likelihood by summing individual contributions
    log_lik = np.sum(log_lik_t)

    return log_lik, pred, filt

