
phip <- function(z) {

    # Based on MATLAB codes provided by Dietmar Bauer, Bielefeld University

    p <- exp(-z^2 / 2) / sqrt(2 * pi)  # Normal pdf

    return(p)

}


phid <- function(z) {

    #
    # taken from Alan Gentz procedures.
    #
    # Based on MATLAB codes provided by Dietmar Bauer, Bielefeld University

    p <- erfc(-z / sqrt(2)) / 2  # Normal cdf

    return(p)

}


logcdf_ME <- function(Zj, Corr_mat) {

    # cdf_ME  : Evaluate approximate log(CDF) according to Mendell Elston
    # Zj      : A column vector of points where CDF is evaluated, of size (len_cdf)
    # Corr_mat: Correlation matrix of size (len_cdf) x (len_cdf)
    #
    # Based on MATLAB codes provided by Dietmar Bauer, Bielefeld University

    source("phip.R") #Univariate normal pdf
    source("phid.R") #Univariate normal cdf
    library(pracma)

    cutoff <- 6

    Zj[Zj > cutoff] <- cutoff
    Zj[Zj < -cutoff] <- -cutoff

    len_cdf <- length(Zj) # dimension of CDF.

    cdf_val <- phid(Zj[1, 1])
    pdf_val <- phip(Zj[1, 1])

    log_res <- log(cdf_val) # perform all calcs in logs.

    if (len_cdf < 2) {
        return(log_res)
    }

    for (jj in 1:(len_cdf - 1)) {

        ajjm1 <- pdf_val / cdf_val

        # Update Zj and Rij
        tZ <- Zj + ajjm1 * Corr_mat[, 1, drop = FALSE]    # update Zj

        R_jj <- Corr_mat[, 1, drop = FALSE] %*% Corr_mat[1, , drop = FALSE]
        tRij <- Corr_mat - R_jj * (ajjm1 + Zj[1, 1]) * ajjm1    # update Rij

        # Convert Rij (i.e. Covariance matrix) to Correlation matrix
        cov2corr <- matrix(sqrt(diag(tRij)), ncol = 1)

        Zj <- tZ / cov2corr
        Corr_mat <- tRij / (cov2corr %*% t(cov2corr))

        # Cutoff those dimensions if they are too low to be evaluated
        cutoff <- 38

        Zj[Zj > cutoff]  <- cutoff
        Zj[Zj < -cutoff] <- -cutoff

        # Evaluate jj's probability
        cdf_val <- phid(Zj[2, 1])
        pdf_val <- phip(Zj[2, 1])

        # Delete unnecessary parts of updated Zj and Corr_mat
        Zj <- Zj[-1, 1, drop = FALSE]
        Corr_mat <- Corr_mat[-1, , drop = FALSE]
        Corr_mat <- Corr_mat[, -1, drop = FALSE]

        # Overall probability
        log_res <- log_res + log(cdf_val)

    }

    return(log_res)

}


dim_red4 <- function(Sigma, Gamma, nu, Delta, cut_tol) {

    # Reduces the dimension of csn
    # according to the correlations of the conditions

    P <- rbind(
      cbind(Sigma, Sigma %*% t(Gamma)),
      cbind(Gamma %*% Sigma, Delta + Gamma %*% Sigma %*% t(Gamma))
    )

    P <- 0.5 * (P + t(P))

    len <- dim(Sigma)[1]
    len2 <- dim(P)[1]


    tryCatch({

        stdnrd <- 1 / sqrt(diag(P))

        if (length(stdnrd) > 1) {
            stdnrd <- diag(stdnrd)
        }

        Pcorr <- abs(stdnrd %*% P %*% stdnrd)

    },
    error = function(codn) {

        ret_list <- list(
          "Sigma" = matrix(0),
          "Gamma" = matrix(0),
          "nu" = matrix(0),
          "Delta" = matrix(0)
        )

        return(ret_list)

    })


    Pcorr <- Pcorr - diag(Inf, nrow = len2)
    Pcorr <- Pcorr[(len + 1):len2, 1:len, drop = FALSE]

    logi2 <- apply(Pcorr, 1, max)

    logi2 <- logi2 < cut_tol

    if (sum(logi2) == 0) {
        logi2 <- 1:(len2 - len)
    } else {
        logi2 <- -1 * (1:(len2 - len))[logi2]
    }

    # Cut unnecessary dimensions
    Gamma <- Gamma[logi2, , drop = FALSE]
    nu <- nu[logi2, , drop = FALSE]
    Delta <- Delta[logi2, , drop = FALSE]
    Delta <- Delta[, logi2, drop = FALSE]

    if (dim(Delta)[1] == 0) {
            Gamma <- matrix(0, nrow = 1, ncol = len)
            nu <- 0
            Delta <- 1
    }

    ret_list <- list(
      "Sigma" = Sigma,
      "Gamma" = Gamma,
      "nu" = nu,
      "Delta" = Delta
    )

    return(ret_list)

}


kalman_csn <- function(
    Y,
    mu_tm1_tm1,
    Sigma_tm1_tm1,
    Gamma_tm1_tm1,
    nu_tm1_tm1,
    Delta_tm1_tm1,
    G,
    R,
    F,
    mu_eta,
    Sigma_eta,
    Gamma_eta,
    nu_eta,
    Delta_eta,
    mu_eps,
    Sigma_eps,
    cut_tol = 1e-2,
    eval_lik = TRUE,
    ret_pred_filt = FALSE,
    logcdfmvna_fct = logcdf_ME
) {

    # -------------------------------------------------------------------------
    # Evaluates log-likelihood value of linear state space model
    # with csn distributed innovations and normally distributed noise:

    #   x[t] = G*x[t-1] + R*eta[t]  [state transition equation]
    #   y[t] = F*x[t]   + eps[t]    [observation equation]
    #   eta[t] ~ CSN(mu_eta, Sigma_eta, Gamma_eta, nu_eta, Delta_eta)
    #                                [innovations, shocks]
    #   eps[t] ~ N(mu_eps,Sigma_eps) [noise, measurement error]


    # Dimensions:
    #   x[t] is (x_nbr by 1) state vector
    #   y[t] is (y_nbr by 1) control vector, i.e. observable variables
    #   eta[t] is (eta_nbr by 1) vector of innovations
    #   eps[t] is (y_nbr by 1) vector of noise (measurement errors)


    # -------------------------------------------------------------------------
    # INPUTS:
    #   - Y:
    #     [y_nbr by obs_nbr]
    #     matrix with data

    #   - mu_tm1_tm1:
    #     [x_nbr by 1]
    #     initial value of location parameter
    #     of CSN distributed states x (does not equal expectation vector
    #     unless Gamma_tm1_tm1=0)

    #   - Sigma_tm1_tm1
    #     [x_nbr by x_nbr]
    #     initial value of scale parameter
    #     of CSN distributed states x (does not equal covariance matrix
    #     unless Gamma_tm1_tm1=0)

    #   - Gamma_tm1_tm1
    #     [skewx_dim by x_nbr]
    #     initial value of first skewness parameter of CSN distributed states x

    #   - nu_tm1_tm1
    #     [skewx_dim by x_nbr]
    #     initial value of second skewness parameter of CSN distributed states x

    #   - Delta_tm1_tm1
    #     [skewx_dim by skewx_dim]
    #     initial value of third skewness parameter of CSN distributed states x

    #   - G:
    #     [x_nbr by x_nbr]
    #     state transition matrix mapping previous states to current states

    #   - R:
    #     [x_nbr by eta_nbr]
    #     state transition matrix mapping current innovations to current states

    #   - F:
    #     [y_nbr by x_nbr]
    #     observation equation matrix mapping current states into current
    #     observables

    #   - mu_eta:
    #     [eta_nbr by 1]
    #     location parameter of CSN distributed innovations eta
    #     (does not equal expectation vector unless Gamma_eta=0)

    #   - Sigma_eta:
    #     [eta_nbr by eta_nbr]
    #     scale parameter of CSN distributed innovations eta
    #     (does not equal covariance matrix unless Gamma_eta=0)

    #   - Gamma_eta:
    #     [skeweta_dim by eta_nbr]
    #     first skewness parameter of CSN distributed innovations eta

    #   - nu_eta:
    #     [skeweta_dim by 1]
    #     second skewness parameter of CSN distributed innovations eta

    #   - Delta_eta:
    #     [skeweta_dim by skeweta_dim]
    #     third skewness parameter of CSN distributed innovations eta

    #   - mu_eps:
    #     [y_nbr by 1]
    #     location parameter of normally distributed measurement errors eps
    #     (equals expectation vector)

    #   - Sigma_eps:
    #     [y_nbr by y_nbr]
    #     scale parameter of normally distributed measurement errors eps
    #     (equals covariance matrix)

    #   - logcdfmvna_fct:
    #     [function name]
    #     name of function with which the log of multivariate normal cdf
    #     is calculated

    #   - cut_tol
    #     [double]
    #     correlation threshold to cut redundant skewness dimensions
    #     as outlined in the paper, if set to 0 no cutting will be done

    #   - eval_lik
    #     [boolean]
    #     TRUE: Carries out log-likelihood computations


    # -------------------------------------------------------------------------
    # OUTPUTS:
    #   - log_lik:
    #     [scalar]
    #     value of log likelihood

    #   - pred:
    #     [structure]
    #     csn parameters
    #     (mu_t_tm1, Sigma_t_tm1, Gamma_t_tm1, nu_t_tm1, Delta_t_tm1)
    #     of predicted states

    #   - filt:
    #     [structure]
    #     csn parameters (mu_t_t,Sigma_t_t,Gamma_t_t,nu_t_t,Delta_t_t)
    #     of filtered states

    # =========================================================================
    # This file is part of the replication files for
    # Guljanov, Mutschler, Trede (2022) - Pruned Skewed Kalman Filter
    # and Smoother: With Application to the Yield Curve

    # Copyright (C) 2022 Gaygysyz Guljanov, Willi Mutschler, Mark Trede

    # This is free software: you can redistribute it and/or modify
    # it under the terms of the GNU General Public License as published by
    # the Free Software Foundation, either version 3 of the License, or
    # (at your option) any later version.

    # This file is distributed in the hope that it will be useful,
    # but WITHOUT ANY WARRANTY without even the implied warranty of
    # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    # GNU General Public License <https://www.gnu.org/licenses/>
    # for more details.


    # Some settings:

    # 'kalman_tol': numerical tolerance for determining the singularity
    # of the covariance matrix of the prediction errors
    # during the Kalman filter
    # (minimum allowed reciprocal of the matrix condition number)
    kalman_tol <- 1e-10

    # 'rescale_prediction_error_covariance': rescales the prediction error
    # covariance in the Kalman filter to avoid badly scaled matrix
    # and reduce the probability of a switch to univariate Kalman filters
    # (which are slower). By default no rescaling is done.
    rescale_prediction_error_covariance <- FALSE
    Omega_singular <- TRUE
    rescale_prediction_error_covariance0 <- rescale_prediction_error_covariance

    # Get dimensions
    dimensions <- dim(F)
    y_nbr <- dimensions[1]
    x_nbr <- dimensions[2]
    obs_nbr <- dim(Y)[2]
    # skeweta_nbr <- dim(Gamma_eta)[1]

    # Initialize some matrices
    mu_eta <- R %*% mu_eta
    Sigma_eta <- R %*% Sigma_eta %*% t(R)
    Gamma_eta <- Gamma_eta %*% solve((t(R) %*% R), t(R))

    Gamma_eta_X_Sigma_eta <- Gamma_eta %*% Sigma_eta
    Delta22_common <- Delta_eta + Gamma_eta_X_Sigma_eta %*% t(Gamma_eta)

    const2pi <- -0.5 * y_nbr * log(2 * pi)

    # Initialize lists to save parameters of predicted/filtered states
    pred <- NaN
    filt <- NaN

    if (ret_pred_filt) {
      pred <- list(
          "mu" = list(),
          "Sigma" = list(),
          "Gamma" = list(),
          "nu" = list(),
          "Delta" = list()
      )

      filt <- list(
          "mu" = list(),
          "Sigma" = list(),
          "Gamma" = list(),
          "nu" = list(),
          "Delta" = list()
      )
    }

    # vector of likelihood contributions
    log_lik_t <- matrix(0, nrow = obs_nbr, ncol = 1)
    log_lik <- -Inf  # default value of log likelihood

    for (t in 1:obs_nbr) {

        # Auxiliary matrices
        Gamma_tm1_tm1_X_Sigma_tm1_tm1 <- Gamma_tm1_tm1 %*% Sigma_tm1_tm1
        Gamma_tm1_tm1_X_Sigma_tm1_tm1_X_GT <- Gamma_tm1_tm1_X_Sigma_tm1_tm1 %*% t(G)


        # Prediction step
        mu_t_tm1 <- G %*% mu_tm1_tm1 + mu_eta

        Sigma_t_tm1 <- G %*% Sigma_tm1_tm1 %*% t(G) + Sigma_eta
        Sigma_t_tm1 <- 0.5 * (Sigma_t_tm1 + t(Sigma_t_tm1)) # ensure symmetry

        invSigma_t_tm1 <- solve(Sigma_t_tm1)

        Gamma_t_tm1 <- (
            rbind(
                Gamma_tm1_tm1_X_Sigma_tm1_tm1_X_GT,
                Gamma_eta_X_Sigma_eta
            )
            %*% invSigma_t_tm1
        )

        nu_t_tm1 <- rbind(nu_tm1_tm1, nu_eta)

        Delta11_t_tm1 <- (
            Delta_tm1_tm1
            + Gamma_tm1_tm1_X_Sigma_tm1_tm1 %*% t(Gamma_tm1_tm1)
                - Gamma_tm1_tm1_X_Sigma_tm1_tm1_X_GT
                %*% invSigma_t_tm1
                %*% t(Gamma_tm1_tm1_X_Sigma_tm1_tm1_X_GT)
        )
        Delta22_t_tm1 <- (
            Delta22_common
            - Gamma_eta_X_Sigma_eta %*% invSigma_t_tm1 %*% t(Gamma_eta_X_Sigma_eta)
        )
        Delta12_t_tm1 <- (
            -Gamma_tm1_tm1_X_Sigma_tm1_tm1_X_GT
            %*% invSigma_t_tm1
                %*% t(Gamma_eta_X_Sigma_eta)
        )

        Delta_t_tm1 <- rbind(
            cbind(Delta11_t_tm1, Delta12_t_tm1),
            cbind(t(Delta12_t_tm1), Delta22_t_tm1)
        )

        Delta_t_tm1 <- 0.5 * (Delta_t_tm1 + t(Delta_t_tm1))  # ensure symmetry

        y_predicted <- F %*% mu_t_tm1 + mu_eps

        prediction_error <- Y[, t, drop = FALSE] - y_predicted

        # Cutting redundant skewness dimension to speed up filtering
        if (cut_tol > 0) {

            red_res <- dim_red4(
                Sigma_t_tm1, Gamma_t_tm1, nu_t_tm1, Delta_t_tm1, cut_tol
            )

            Sigma_t_tm1 <- red_res$Sigma
            Gamma_t_tm1 <- red_res$Gamma
            nu_t_tm1 <- red_res$nu
            Delta_t_tm1 <- red_res$Delta

        }

        # Kalman gains
        Omega <- F %*% Sigma_t_tm1 %*% t(F) + Sigma_eps
        Omega <- 0.5 * (Omega + t(Omega))  # ensure symmetry

        badly_conditioned_Omega <- FALSE

        sig <- matrix(sqrt(diag(Omega)), ncol = 1)

        if (rescale_prediction_error_covariance) {

            if (any(diag(Omega) < kalman_tol)
            || rcond(Omega / (sig %*% t(sig))) < kalman_tol) {

                badly_conditioned_Omega <- TRUE

                warning("badly_conditioned_Omega")

            }

        } else {

            if (rcond(Omega) < kalman_tol) {

                if (any(diag(Omega) < kalman_tol)
                || rcond(Omega / (sig %*% t(sig))) < kalman_tol) {

                    badly_conditioned_Omega <- TRUE

                    warning("badly_conditioned_Omega")

                } else {

                    rescale_prediction_error_covariance <- 1
                    warning(paste(
                        "set rescale_prediction_error_covariance",
                        "to 1"
                    ))

                }

            }

        }

        if (badly_conditioned_Omega) {

            if (!all(abs(Omega) < kalman_tol)) {
                # Use univariate filter
                # (will remove observations with zero variance prediction error)
                warning("Univariate filter not yet for CSN")
            } else {
                # Pathological case, discard draw
                warning("Discard draw due to badly_conditioned_Omega")
            }

            log_lik <- NaN

            return(list("log_lik" = log_lik, "pred" = pred, "filt" = filt))

        }

        Omega_singular <- FALSE

        if (rescale_prediction_error_covariance) {

            log_detOmega <- log(det(Omega / (sig %*% t(sig)))) + 2 * sum(log(sig))

            invOmega <- solve(Omega / (sig %*% t(sig))) / (sig %*% t(sig))

            rescale_prediction_error_covariance <- rescale_prediction_error_covariance0

        } else {

            log_detOmega <- log(det(Omega))

            invOmega <- solve(Omega)

        }

        K_Gauss <- Sigma_t_tm1 %*% t(F) %*% invOmega
        K_Skewed <- Gamma_t_tm1 %*% K_Gauss


        # log-likelihood contributions
        if (eval_lik) {

            # The conditional distribution of y[t] given y[t-1] is:
            # (y[t] | y[t-1]) ~ CSN(mu_y, Sigma_y, Gamma_y, nu_y, Delta_y)
            # = (
            #    mvncdf(Gamma_y * (y[t] - mu_y), nu_y, Delta_y)
            #    / mvncdf(0, nu_y, Delta_y + Gamma_y * Sigma_y * t(Gamma_y))
            #    * mvnpdf(y[t], mu_y, Sigma_y)
            # )
            #
            # where:
            #  mu_y    = F %*% mu_t_tm1 + mu_eps = y_predicted
            #  Sigma_y = F %*% Sigma_t_tm1 %*% t(F) + Sigma_eps = Omega
            #  Gamma_y = Gamma_t_tm1 %*% Sigma_t_tm1 %*% t(F)
            #            %*% inv(F %*% Sigma_t_tm1 %*% t(F) + Sigma_eps) = K_Skewed
            #  nu_y    = nu_t_tm1
            #  Delta_y = Delta_t_tm1
            #            + Gamma_t_tm1 %*% Sigma_t_tm1 %*% t(Gamma_t_tm1)
            #            - Gamma_t_tm1 %*% Sigma_t_tm1 %*% t(F)
            #              %*% inv(F %*% Sigma_t_tm1 %*% t(F))
            #              %*% F %*% Sigma_t_tm1 %*% t(Gamma_t_tm1)
            #            + (Gamma_t_tm1 %*% Sigma_t_tm1 %*% t(F)
            #                 %*% inv(F*Sigma_t_tm1*t(F))
            #               - Gamma_t_tm1 %*% Sigma_t_tm1 %*% t(F)
            #                 %*% inv(F %*% Sigma_t_tm1 %*% t(F) + Sigma_eps))
            #              %*% F %*% Sigma_t_tm1 %*% t(Gamma_t_tm1)
            #          = Delta_t_tm1 + (Gamma_t_tm1 - K_Skewed %*% F)
            #            %*% Sigma_t_tm1 %*% t(Gamma_t_tm1)

            Delta_y <- (
                Delta_t_tm1
                + (Gamma_t_tm1 - K_Skewed %*% F) %*% Sigma_t_tm1 %*% t(Gamma_t_tm1)
            )
            Delta_y <- 0.5 * (Delta_y + t(Delta_y)) # ensure symmetry

            # evaluate Gaussian cdfs, i.e.
            #  - bottom one:
            #    mvncdf(0, nu_y, Delta_y + Gamma_y %*% Sigma_y %*% t(Gamma_y))
            #  - top one:
            #    mvncdf(Gamma_y %*% (y[t] - mu_y), nu_y, Delta_y)
            cdf_bottom_cov <- Delta_y + K_Skewed %*% Omega %*% t(K_Skewed)
            cdf_bottom_cov <- 0.5 * (cdf_bottom_cov + t(cdf_bottom_cov))

            if (!identical(logcdfmvna_fct, logcdf_ME)) {
                stop(paste(
                    "Only logcdf_ME has been implemented",
                    "for normal cdf evaluation"
                ))
            }

            # Evaluate bottom cdf
            tmp_cov2corr <- 1 / sqrt(diag(cdf_bottom_cov))

            if (length(tmp_cov2corr) > 1) {
                tmp_cov2corr <- diag(tmp_cov2corr)
            }

            cdf_bottom_cov <- tmp_cov2corr %*% cdf_bottom_cov %*% tmp_cov2corr
            cdf_bottom_cov <- 0.5 * (cdf_bottom_cov + t(cdf_bottom_cov))

            shape_bottom <- dim(cdf_bottom_cov)
            if (shape_bottom[1] + shape_bottom[2] > 0) { # if not empty
                log_gaussian_cdf_bottom <- logcdfmvna_fct(
                    -tmp_cov2corr %*% nu_t_tm1, cdf_bottom_cov
                )
            } else {
                log_gaussian_cdf_bottom <- 0
            }

            # Evaluate top cdf
            tmp_cov2corr <- 1 / sqrt(diag(Delta_y))

            if (length(tmp_cov2corr) > 1) {
                tmp_cov2corr <- diag(tmp_cov2corr)
            }

            cdf_top_cov <- tmp_cov2corr %*% Delta_y %*% tmp_cov2corr
            cdf_top_cov <- 0.5 * (cdf_top_cov + t(cdf_top_cov))

            shape_top <- dim(cdf_top_cov)
            if (shape_top[1] + shape_top[2] > 0) {
                log_gaussian_cdf_top <- logcdfmvna_fct(
                    tmp_cov2corr %*% (K_Skewed %*% prediction_error - nu_t_tm1),
                    cdf_top_cov
                )
            } else {
                log_gaussian_cdf_top <- 0
            }

            # evaluate Gaussian pdf
            # log_gaussian_pdf = log(mvnpdf(Y[:, t], y_predicted, Omega))
            log_gaussian_pdf <- (const2pi
            - 0.5 * log_detOmega - 0.5 * t(prediction_error) %*% invOmega %*% prediction_error)

            log_lik_t[t] <- (
                -log_gaussian_cdf_bottom
                + log_gaussian_pdf
                + log_gaussian_cdf_top
            )

            if (is.nan(log_lik_t[t])) {
                stop(sprintf("Likelihood contribution is NaN at iteration = %i!", t))
            }

        }


        # Filtering step
        mu_t_t <- mu_t_tm1 + K_Gauss %*% prediction_error

        Sigma_t_t <- Sigma_t_tm1 - K_Gauss %*% F %*% Sigma_t_tm1

        Gamma_t_t <- Gamma_t_tm1

        nu_t_t <- nu_t_tm1 - K_Skewed %*% prediction_error

        Delta_t_t <- Delta_t_tm1

        Sigma_t_t <- 0.5 * (Sigma_t_t + t(Sigma_t_t))  # ensure symmetry
        Delta_t_t <- 0.5 * (Delta_t_t + t(Delta_t_t))  # ensure symmetry

        # assign for next time step
        mu_tm1_tm1 <- mu_t_t
        Sigma_tm1_tm1 <- Sigma_t_t
        Gamma_tm1_tm1 <- Gamma_t_t
        nu_tm1_tm1 <- nu_t_t
        Delta_tm1_tm1 <- Delta_t_t

        if (ret_pred_filt) {

            # save the parameters of the predicted and filtered csn states
            pred$mu[[t]] <- mu_t_tm1
            pred$Sigma[[t]] <- Sigma_t_tm1
            pred$Gamma[[t]] <- Gamma_t_tm1
            pred$nu[[t]] <- nu_t_tm1
            pred$Delta[[t]] <- Delta_t_tm1

            filt$mu[[t]] <- mu_t_t
            filt$Sigma[[t]] <- Sigma_t_t
            filt$Gamma[[t]] <- Gamma_t_t
            filt$nu[[t]] <- nu_t_t
            filt$Delta[[t]] <- Delta_t_t

        }

    }

    if (Omega_singular) {
        stop(paste(
            "The variance of the forecast error",
            "remains singular until the end of the sample"
        ))
    }

    # Compute log-likelihood by summing individual contributions
    log_lik <- sum(log_lik_t)

    return(list("log_lik" = log_lik, "pred" = pred, "filt" = filt))

}
