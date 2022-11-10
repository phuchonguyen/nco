//
// Product parametrization model
//

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> N; // number of NCO
  int<lower=0> M; // number of exposure settings
  vector[N] Y[M]; // estimated ATE
  vector<lower=0>[N] sigma[M]; // s.e. of estimated ATE
}

// The parameters accepted by the model.
parameters {
  // biases contributed by the association between 
  // the NCO and unobserved confounder
  vector[N] delta;
  // biases contributed by the association between
  // the treatment and the unobserved confounder
  vector[M] eta;
  real mu; // global mean for deltas
  real<lower=0> tau;
  real zeta; // global mean for etas
  real<lower=0> nu;
}

transformed parameters {
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  for (i in 1:M)
    Y[i] ~ normal(delta * eta[i], sigma[i]);
  delta ~ normal(mu, tau);
  eta ~ normal(zeta, nu);
  mu ~ normal(1, 1);
  tau ~ gamma(2, 2);
  zeta ~ normal(1, 1);
  nu ~ gamma(2, 2);
}

generated quantities {
  // posterior predictions
  vector[N] Y_rep[M];
  for (i in 1:M)
    Y_rep[i] = delta * eta[i];
}















