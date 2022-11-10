//
// Shared global mean for all exposure setting
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
  // unobserved biases is the same for one NCO across exposure settings
  vector[N] delta;
  // mean of unobserved biases for all exposure settings
  real mu;
  // s.d. of unobserved bias
  real<lower=0> tau;
}

transformed parameters {
}

// The model to be estimated.
model {
  for (i in 1:M)
    Y[i] ~ normal(delta, sigma[i]);
  delta ~ normal(mu, tau);
  mu ~ normal(1, 1);
  tau ~ gamma(2, 2);
}

generated quantities {
  // posterior predictions
  vector[N] Y_rep[M];
  for (i in 1:M)
    Y_rep[i] = delta;
}















