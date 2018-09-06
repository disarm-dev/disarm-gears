// Gaussian process - Beta regression
// ----------------------------------

functions {

  /**
   * Exponentiated Quadratic Kernel
   *
   * @param X : Inputs
   * @param cov_var : Variance parameter
   * @param cov_length : Vector of kernel lengthscales
   * @param delta : jitter
   *
   * @return Exponentiated Quadratic Kernel K(X, X)
   */
   //
  matrix k_exp_quad_ard(vector[] X, real cov_var, vector cov_length, real delta) {

    int n = size(X);
    matrix[n, n] K;

    for (i in 1:(n-1)) {
      K[i, i] = cov_var + delta;
      for (j in (i +1):n) {
        K[i, j] = cov_var * exp(-.5 * dot_self((X[i] - X[j]) ./ cov_length));
        K[j, i] = K[i, j];
      }
    }
    K[n, n] = cov_var + delta;

    return K;

  }

}


data {

  int<lower=1> n_data;
  int<lower=1> n_dims;
  vector[n_data] y_data;
  vector[n_dims] X_data[n_data];
  vector[n_data] mu_data;

}


transformed data {

  real delta = 1e-6;

}


parameters {

  real<lower=0> theta;
  real<lower=0> cov_var;
  vector<lower=0>[n_dims] cov_length;
  vector[n_data] eta;

}


model {

  matrix [n_data, n_data] K = k_exp_quad_ard(X_data, cov_var, cov_length, delta);
  matrix[n_data, n_data]  L = cholesky_decompose(K);

  vector[n_data] f = mu_data + L * eta;
  vector[n_data] phi = inv_logit(f);

  theta ~ normal(0, 1);
  cov_var ~ normal(0, 1);
  for (i in 1:n_dims) cov_length[i] ~ inv_gamma(2, 1); //5,5
  eta ~ normal(0, 1);
  for (i in 1:n_data) y_data[i] ~  beta(phi[i]*theta, (1-phi[i])*theta);

}




