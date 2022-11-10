//
// Independent models for each exposure setting
//

// The input data
data {
  int<lower=0> N; // number of NCO
  int<lower=0> M; // number of exposure settings
  vector[N] Y[M]; // estimated ATE
  vector<lower=0>[N] sigma[M]; // s.e. of estimated ATE
}

// The parameters accepted by the model.
parameters {
  // unobserved biases
  vector[N] delta[M];
  // mean of unobserved bias for each exposure setting
  real mu[M];
  // s.d. of unobserved bias
  real<lower=0> tau[M];
}

transformed parameters {
}

// The model to be estimated.
model {
  for (i in 1:M) {
    Y[i] ~ normal(delta[i], sigma[i]);
    delta[i] ~ normal(mu[i], tau[i]);
    mu[i] ~ normal(1, 1);
    tau[i] ~ gamma(2, 2);
  }
}

generated quantities {
  // posterior predictions
  vector[N] Y_rep[M];
  for (i in 1:M)
    Y_rep[i] = delta[i];
}















