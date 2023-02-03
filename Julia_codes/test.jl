eval_p = [0.2, 0.3, -0.8]

Corr_mat = [1.0 0.2 0.3; -0.4 1.0 -0.9; -0.5 0.8 1.0]
Corr_mat = 0.5 * (Corr_mat + Corr_mat')

include("skalman_filter.jl")
logcdf_ME(eval_p, Corr_mat)


breakpoint(abs)

@run logcdf_ME(eval_p, Corr_mat)


function add(a, b)

    z = a + b

    return z

end

function subtract(a::T) where T<:Real

    z = a - 2 

    return z

end


function disp(fun, a, b)

    if fun == add
        print(fun(a, b))
    elseif fun == subtract
        print(fun(a))
        print(fun(b))
    else
        print("Give a damn correct function")
    end

end

c = disp(add, 3, 2)
c = disp(subtract, 3, 2)

try
    sqrt(-2)
catch
    print("Aha")
end





using MAT
using LinearAlgebra

file = matopen("../../Skewed_lin_DSGE/Codes/test_data.mat")
data = read(file, "y_mat")
data = copy(transpose(data))
close(file)

# include("skalman_filter.jl")
# kalman_csn(
#     Y=data,
#     mu_tm1_tm1=[0.00],
#     Sigma_tm1_tm1=[10.00;;],
#     Gamma_tm1_tm1=[0.00;;],
#     nu_tm1_tm1=[0.00],
#     Delta_tm1_tm1=[1.00;;],
#     G=[0.50;;],
#     R=[1.00;;],
#     F=[0.30;;],
#     mu_eta=[0.00],
#     Sigma_eta=[1.00;;],
#     Gamma_eta=[0.50;;],
#     nu_eta=[0.00],
#     Delta_eta=[1.00;;],
#     mu_eps=[0.00],
#     Sigma_eps=[1.00;;],
#     cut_tol=0.1,
#     eval_lik=true,
#     ret_pred_filt=false,
#     logcdfmvna_fct=logcdf_ME,
# )


include("skalman_filter.jl")
liki, pred, filt = kalman_csn(
    Y=data,
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
    Gamma_eta=[0.50 1.2; -1.8 2],
    nu_eta=zeros(2),
    Delta_eta=Matrix(1.00I, 2, 2),
    mu_eps=zeros(2),
    Sigma_eps=Matrix(1.00I, 2, 2),
    cut_tol=0.1,
    eval_lik=true,
    ret_pred_filt=true,
    logcdfmvna_fct=logcdf_ME,
);
