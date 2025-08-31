# Pruned Skewed Kalman filter codes in R, Python, Julia and MATLAB
This is the documentation for the 
Pruned Skewed Kalman filter(PSKF) implementation 
in different languages.

This filter was introduced in the paper 
"Pruned Skewed Kalman Filter and Smoother: 
With Applications to the Yield Curve"
by Gaygysyz Guljanov, Willi Mutschler and Mark Trede (2022).
The complete replications codes for this paper is given at
[Willi Mutschler's github repository](https://github.com/wmutschl/pruned-skewed-kalman-paper).


In this repo, the following functions are present for each language:
- `phip()`: Evaluates the univariate 
Normal probability density function
- `phid()`: Evaluates the univariate 
Normal cumulative density function
- `logcdf_ME()`: Evaluates the multivariate 
Normal cumulative density function
- `dim_red4()`: Reduces the skewness dimension of csn 
based on correlations
- `kalman_csn()`: Evaluates the log-likelihood using PSKF

`phip()`, `phid()` and `logcdf_ME()` functions are slightly modified versions 
of `phip()`, `phid()` and `mcdfmvna_ME()` functions by @dbauer72, Dietmar Bauer, see [the repo](https://github.com/dbauer72/MaCML-MATLAB-Code).

All the functions above are used by `kalman_csn()`.


<br>

Data generating process is the following state-space model:

```math
    Y_t = F X_t + \epsilon_t, ~~~ 
    \epsilon_t \sim N(\mu_\epsilon, \Sigma_\epsilon), ~~~
    \rightarrow \text{measurement equation}
```
```math
    X_t = G X_{t-1} + R \eta_t, ~~~
    \eta_t \sim CSN(
        \mu_\eta, \Sigma_\eta, \Gamma_\eta, \nu_\eta, \Delta_\eta
    ), ~~~
    \rightarrow \text{state equation}
```

$Y_t$ is **y_nbr** dimensional observable random variable. <br>
$X_t$ is **x_nbr** dimensional latent random (state) variable. <br>
$\epsilon_t$ is **y_nbr** dimensional Normal random variable. <br>
$\eta_t$ is **eta_nbr** dimensional CSN random variable
with skewness dimension of **skeweta_dim**. <br>
Time $t$ runs up to **obs_nbr**.


<br>

**Input arguments** of `kalman_csn()` are roughly as follows:
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
  initial value of first skewness parameter
  of CSN distributed states x
- nu_tm1_tm1      
  [skewx_dim by x_nbr]           
  initial value of second skewness parameter 
  of CSN distributed states x
- Delta_tm1_tm1   
  [skewx_dim by skewx_dim]       
  initial value of third skewness parameter 
  of CSN distributed states x
- G:              
  [x_nbr by x_nbr]               
  state transition matrix mapping previous states to current states
- R:              
  [x_nbr by eta_nbr]             
  state transition matrix mapping current innovations 
  to current states
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

<br>

Please report any bugs or weird behavior of the functions.
