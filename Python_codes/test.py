# The main function

# eval_p = np.array([[0.2, 0.3, -0.8]]).T

# Corr_mat = np.array([[1.0, 0.2, 0.3], [-0.4, 1.0, -0.9], [-0.5, 0.8, 1.0]])
# Corr_mat = 0.5 * (Corr_mat + Corr_mat.T)


def main():

    import skalman_filter as skf
    import warnings
    import os
    import numpy as np
    import scipy
    from scipy import special

    # print(logcdf_ME(eval_p, Corr_mat))
    data = scipy.io.loadmat("../../Skewed_lin_DSGE/Codes/test_data.mat")
    data = data["y_mat"]

    # log_lik, pred_val, filt_val = skf.kalman_csn(
    #     Y=data.T,
    #     mu_tm1_tm1=np.array([[0]]),
    #     Sigma_tm1_tm1=np.array([[10]]),
    #     Gamma_tm1_tm1=np.array([[0]]),
    #     nu_tm1_tm1=np.array([[0]]),
    #     Delta_tm1_tm1=np.array([[1]]),
    #     G=np.array([[0.5]]),
    #     R=np.array([[1]]),
    #     F=np.array([[0.3]]),
    #     mu_eta=np.array([[0]]),
    #     Sigma_eta=np.array([[1]]),
    #     Gamma_eta=np.array([[0.5]]),
    #     nu_eta=np.array([[0]]),
    #     Delta_eta=np.array([[1]]),
    #     mu_eps=np.array([[0]]),
    #     Sigma_eps=np.array([[1]]),
    #     cut_tol=0.1,
    #     eval_lik=True,
    #     ret_pred_filt=True,
    #     logcdfmvna_fct=skf.logcdf_ME,
    # )

    log_lik, pred_val, filt_val = skf.kalman_csn(
        Y=data.T,
        mu_tm1_tm1=np.array([[0], [0]]),
        Sigma_tm1_tm1=np.array([[10, 0], [0, 10]]),
        Gamma_tm1_tm1=np.zeros((2, 2)),
        nu_tm1_tm1=np.zeros((2, 1)),
        Delta_tm1_tm1=np.eye(2),
        G=np.array([[0.7, -0.5], [0.3, 0.9]]),
        R=np.eye(2),
        F=np.array([[0.3, 1], [2, -5]]),
        mu_eta=np.zeros((2, 1)),
        Sigma_eta=np.eye(2),
        Gamma_eta=np.array([[0.5, 1.2], [-1.8, 2]]),
        nu_eta=np.zeros((2, 1)),
        Delta_eta=np.eye(2),
        mu_eps=np.zeros((2, 1)),
        Sigma_eps=np.eye(2),
        cut_tol=0.1,
        eval_lik=True,
        ret_pred_filt=True,
        logcdfmvna_fct=skf.logcdf_ME,
    )

    print(log_lik)

    print(pred_val["nu"][29])
    print(filt_val["nu"][29])


if __name__ == "__main__":
    main()
