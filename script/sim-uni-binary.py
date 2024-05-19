# Simulation settings similar to Gruber & Tchetgen 2017
# Generate ONE i.i.d Normal unmeasured confounders U
# Number of NCOs: 30
# Number of subjects in NCO study: 10,000
# Number of Y outcomes: 1000
# Outcome proportions of all NCOs and Ys are around 0.5
# Binary treatment A
# Bias threshold is 1.1 on the risk ratio scale
# Probability threshold is 0.95
# Generate the data so that the true bias distribution is
# NOT less than 1.1 w.p. 0.95.
from pymc.gp.util import plot_gp_dist
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from models import get_mean_model, run_polynomial
from helpers import get_boostrap_rr
import arviz as az
import numpy as np
from scipy.stats import bernoulli
from scipy.special import expit
import math
import pymc as pm
print(f"Running on PyMC v{pm.__version__}")

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Simulate data
# Set number of observations
n = 10000
# Set number of unobserved confounders
p_u = 1
# Set number of treatment groups
n_a = 2
# Set number of scenarios/treatment contrasts
K = math.comb(n_a, 2)
# Define a sample of outcomes for validation
n_y = 1000
# Define number of NCOs
I = 30
# Set parameters for fitting models
n_samples = 2000
n_tune = 5000
n_chains = 2
# Number of repetition of the simulation
n_rep = 100  # TODO
# Set the number of bootstrap samples
n_bootstrap = 100
# Parameters for decision rule to proceed with the study or not
bias_threshold = np.log(1.1)
prob_bound = 0.95

# Generate latent space regression coefficients of U on A
# np.random.uniform(0.2, 0.5, size=p_u)  # effect of U on treatment assigment A
alpha_u = np.array([0.5])
print(alpha_u)

# Generate unobserved confounder from standard normal
# Generate another dataset with 1M samples to approximate the truth
U_1m = np.random.normal(size=(1000000, p_u))

# Define the treatment groups by discretizing the latent A
A_1m = np.matmul(U_1m, alpha_u)

# For binary treatment, use logistic model
A_obs_1m = bernoulli.rvs(p=expit(A_1m))

# Define the effects of U on the Y in latent space
beta_u_low = 0.5
beta_u_high = 0.75
theta_y = np.random.uniform(beta_u_low, beta_u_high, size=(p_u, n_y))

# Randomly zero-out the effects w.p. phi_delta
phi_delta = 0.3
theta_y *= np.random.choice([0, 1], size=(p_u, n_y),
                            p=[phi_delta, 1-phi_delta])

# Randomly shuffle the sign of the effects with probability phi_m
phi_m = 0.7
theta_y *= np.random.choice([-1, 1], size=(p_u, n_y),
                            p=[phi_delta, 1-phi_delta])

# Generate n_y binary outcomes with treatment effects in [0, 1]
beta = np.zeros(n_y)  # np.random.uniform(0, 1, size=(n_y))
preds_0 = np.column_stack((U_1m, np.zeros(len(U_1m))))
preds_1 = np.column_stack((U_1m, np.ones(len(U_1m))))
b_tmps = np.row_stack((theta_y, beta))
# counterfactual Y under treatment
Y_1 = bernoulli.rvs(p=expit(np.matmul(preds_1, b_tmps)))  # 1M x n_y matrix
# counterfactual Y under control
Y_0 = Y_1  # bernoulli.rvs(p=expit(np.matmul(preds_0, b_tmps)))

# Calculate true treatment effects and true biases on RR scale
# true effect size using counterfactual outcomes
# np.mean(Y_1[i] - Y_0[i], axis=0)
effect_Y_true = np.mean(Y_1, axis=0) / np.mean(Y_0, axis=0)
# observed effect size that is biased
bias_Y_true = np.mean(Y_1[A_obs_1m == 1, :], axis=0) / \
    np.mean(Y_0[A_obs_1m == 0, :], axis=0)
# calculate the bias on the log RR scale: lRRbias = lRRobs - lRRtrue is equivalent to exp(lbias) = RRobs / RRtrue
bias_Y_true /= effect_Y_true
# turn into an n_y x 1 matrix
bias_Y_true = bias_Y_true.transpose()
# save the true bias distribution
np.save('output/sim-uni-binary/true-lrr-bias-' +
        str(I) + '.npy', np.log(bias_Y_true))

# Fit the mean model
# Store the true decision result based on the true distribution of A-Y biases
mean_total = np.empty((n_samples, n_rep))
poly_total = np.empty((n_samples, n_rep))

for i in range(n_rep):
    # For reproducibility
    np.random.seed(RANDOM_SEED + i + 50)

    # Generate unobserved confounder from standard normal
    U = np.random.normal(size=(n, p_u))

    # Define the treatment groups by discretizing the latent A
    A = np.matmul(U, alpha_u)
    A_obs = bernoulli.rvs(p=expit(A))

    # Define the effects of U on the NCOs in latent space
    theta_u = np.random.uniform(beta_u_low, beta_u_high, size=(p_u, I))

    # Randomly zero-out the effects w.p. phi_delta
    theta_u *= np.random.choice([0, 1], size=(p_u, I),
                                p=[phi_delta, 1-phi_delta])

    # Randomly shuffle the sign of the effects with probability phi_m
    theta_u *= np.random.choice([-1, 1], size=(p_u, I), p=[phi_m, 1-phi_m])

    # Generate binary NCOs
    N = bernoulli.rvs(p=expit(np.matmul(U, theta_u)))

    # Calculate NCO bias estimates
    bias_N = np.mean(N[A_obs == 1, :], axis=0) / \
        np.mean(N[A_obs == 0, :], axis=0)
    # Reshape into a (n, 1) matrix
    bias_N = np.expand_dims(bias_N, 1)

    # Calculate NCO bias s.e. using bootstrapping
    # Create random sample with replacement of row indinces
    b_indices = np.random.choice(range(n), size=(n_bootstrap, n), replace=True)
    # Apply statistics to bootstrapped samples
    b_bias = np.apply_along_axis(
        get_boostrap_rr, axis=1, arr=b_indices, ncos=N, treatment=A_obs, n=n_a)
    # Standard deviation on the log risk ratio scale
    b_bias = np.log(b_bias)
    # Calculate s.e. of statistics in bootstrap samples
    se_bias_N = np.std(b_bias, axis=0, ddof=1)

    # Mean model bias on the log RR scale
    # Set informed hyperparameters on prior for delta and m
    model_hp = get_mean_model(np.log(bias_N), se_bias_N,
                              alpha_d=1, beta_d=1,  # 1/2 of estiamtes are not statistically significant
                              alpha_m=1, beta_m=1,  # 1/2 of estimates are positive
                              # variance in data is around 0.02, Pr(sigma^2 <= 0.02) = 0.95
                              # alpha_sigma=3.35, beta_sigma=0.02,
                              nu=0, tau=0.5**2)  # 95percent that bias in RR between 0.36-2.7
    # MCMC sampling
    with model_hp:
        trace_hp = pm.sample(n_samples, tune=n_tune, chains=n_chains,
                             target_accept=0.97, random_seed=RANDOM_SEED)
    # Get posterior predictive of the magnitude of the bias
    # a.k.a. predict the absolute value of the bias in Y
    # For each MCMC iteration:
    # 1. simulate \delta_Y| \phi_d
    # 2. S_Yk|\delta_Y, \beta_k, \sigma_k, m_Y = 1 for all K
    num_samps = n_samples
    samps_hp = az.extract(trace_hp, var_names=[
                          "phi_d", 'beta', 'sigma'], num_samples=num_samps).to_dataframe()
    samps_hp = samps_hp.drop(['draw', 'chain'], axis=1)
    samps_hp = samps_hp.reset_index()
    samps_hp.sort_values(by=['chain', 'draw', 'setting'], inplace=True)
    samps_hp['delta_y'] = np.repeat(np.random.binomial(
        1, samps_hp.groupby(['draw', 'chain'])['phi_d'].min()), K)
    samps_hp['slap_y'] = np.random.normal(
        np.abs(samps_hp['beta']), np.sqrt(samps_hp['sigma']), len(samps_hp))
    samps_hp['slap_y'] = np.abs(samps_hp.slap_y)
    samps_hp['bias_posterior'] = (1-samps_hp['delta_y'])*samps_hp['slap_y']

    # # Calculate the posterior predictive probability that the magnitude of the
    # # bias is less than a bias threshold on RR scale
    # pp = np.mean(samps_hp.bias_posterior <= bias_threshold)
    # # Proceed with the study if res is more than a probability bound
    # 1*(pp >= prob_bound)
    # proceed_total[i+1] = pp

    # Save the result to a binary file
    mean_total[:, i] = samps_hp.bias_posterior
    np.save('output/sim-uni-binary/mean-lrr-bias-' +
            str(I) + '.npy', mean_total)

    # Model the bias using Heteroskedastic GP
    # model_hetero, trace_hetero = run_hetero_gp(np.log(bias_N),
    #                                            se_bias_N,
    #                                            # input on the (0, 1) scale
    #                                            x=np.mean(
    #                                                N, axis=0).reshape(-1, 1),
    #                                            #    alpha_d=1, beta_d=1, # 1/2 of estiamtes are not statistically significant
    #                                            #    alpha_m=1, beta_m=1, # 1/2 of estimates are positive
    #                                            # phi_d=0.01, phi_m=0.99,
    #                                            sample=n_samples//2,
    #                                            tune=5000//2,
    #                                            chain=2,
    #                                            x_star=np.array(
    #                                                [0.5]).reshape(-1, 1),
    #                                            random_seed=RANDOM_SEED, target_accept=0.95, cores=2,
    #                                            #    sigma_v=0.3
    #                                            )

    # mu_star = az.extract(trace_hetero.posterior_predictive,
    #                      var_names='mu_star').transpose("sample", ...)
    # f_star = az.extract(trace_hetero.posterior_predictive,
    #                     var_names='f_star').transpose("sample", ...)

    # Save the result to a binary file
    # gp_total[:, i] = mu_star[:, 0]
    # np.save('output/gp-050-lrr-bias-' + str(I) + '.npy', gp_total)
    model_poly, trace_poly = run_polynomial(
        np.log(bias_N),
        se_bias_N,
        np.mean(N, axis=0).reshape(-1, 1),
        sample=1000, tune=3000, chain=2,
        x_star=np.array([0.5]).reshape(-1, 1),
        random_seed=123, target_accept=0.8, cores=2)
    mu_star = az.extract(trace_poly.posterior_predictive,
                         var_names='mu_star').transpose("sample", ...)

    # Save the result to a binary file
    poly_total[:, i] = mu_star[:, 0]
    np.save('output/sim-uni-binary/poly-lrr-bias-' +
            str(I) + '.npy', poly_total)

# Plot results

# Plot observed NCO estimates
# Function to add error bars


# def add_error_bars(x, y, error, **kwargs):
#     data = kwargs.pop("data")
#     ax = plt.gca()
#     for i, (_, row) in enumerate(data.iterrows()):
#         ax.errorbar(row['bias'], row['nco'], xerr=1.96 *
#                     row['se'], fmt='none', ecolor='black', capsize=3)
# # Function to replace title


# def replace_subtitle(x, y, title='', **kwargs):
#     data = kwargs.pop("data")
#     ax = plt.gca()
#     ax.set_title(title)


# # Reshape data to long format
# data_columns = ['setting_' + str(i) for i in range(K)]
# data_index = ['nco_' + str(i) for i in range(I)]
# # bias is on RR scale
# data = pd.DataFrame(np.log(bias_N), columns=data_columns)
# data['nco'] = data_index
# data_long = pd.melt(data,
#                     id_vars=['nco'],
#                     value_vars=data_columns,
#                     var_name='setting', value_name='bias')
# data_long['se'] = se_bias_N.flatten('F')
# # Initialize a facet grid plot
# g = sns.FacetGrid(data_long, col='setting', height=5, aspect=1)
# # Create the scatter plot of the simulated data
# g.map(sns.scatterplot, 'bias', 'nco', alpha=1, color='black')
# # Add error bars faceted by 'type'
# g.map_dataframe(add_error_bars, 'bias', 'nco', 'se')
# # Remove title
# g.map_dataframe(replace_subtitle, 'bias', 'nco', '')
# # Add verticle line at 0.0
# g.refline(x=0.0)
# # Change axis labels
# g.set_axis_labels("Estimated Log Risk Ratio", "Negative Control Outcomes")
# # Change x ticks label
# g.set_yticklabels(['NCO ' + str(i+1) for i in range(I)])
# # Save plot to pdf
# # plt.savefig("data-30nco.pdf", format="pdf", bbox_inches="tight")
# # Show the plot
# plt.show()

# # Plot the posterior of the GP mean and GP variance
# # ls_p = az.extract(trace_hetero, var_names='ls').transpose("sample", ...)
# # plt.hist(ls_p)
# # m_ppc = az.extract(trace_hetero, var_names='m').transpose("sample", ...)
# # v_ppc = az.extract(trace_hetero, var_names='v').transpose("sample", ...)
# # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
# # plot_gp_dist(ax1, np.abs(m_ppc), np.mean(N, axis=0))
# # plot_gp_dist(ax2, v_ppc, np.mean(N, axis=0))
# # ax1.set_title('Posterior predictive of the mean function')
# # ax2.set_title('Posterior predictive of the variance function')
# # plt.show()

# # Plot posterior predictive distributions
# # Define truth
# truth = np.abs(np.log(bias_Y_true))
# # Define weights so that the histogram bar at zero reflects the probability that the bias is zero
# weights = np.ones(num_samps)/float(num_samps)
# binwidth = 0.01
# # Create a FacetGrid with histograms
# g = sns.FacetGrid(samps_hp, col='setting', height=5, aspect=1)
# g.map(plt.hist, 'bias_posterior',
#       bins=np.arange(min(samps_hp.bias_posterior), max(
#           samps_hp.bias_posterior) + binwidth, binwidth),
#       alpha=0.5,
#       weights=weights,
#       label='Mean PPD'
#       )
# # g.map_dataframe(replace_subtitle, 'bias_posterior', 'bias_posterior', '')
# # Add density of true biases and GP biases
# for ax, setting in zip(g.axes.flat, g.col_names):
#     ax.hist(np.abs(mu_star[:, 0]), weights=np.ones(len(mu_star))/float(len(mu_star)),
#             bins=np.arange(min(np.abs(mu_star[:, 0])), max(
#                 np.abs(mu_star[:, 0])) + binwidth, binwidth),
#             color='green', alpha=0.3,
#             label='Polynomial at x=0.5')
#     # ax.hist(np.abs(mu_star[:, 1]), weights=np.ones(len(mu_star))/float(len(mu_star)),
#     #         bins=30, color='magenta', alpha=0.3,
#     #         label='GP PPD x=0.5')
#     # ax.hist(np.abs(mu_star[:, 2]), weights=np.ones(len(mu_star))/float(len(mu_star)),
#     #         bins=30, color='red', alpha=0.3,
#     #         label='GP PPD x=0.9')
#     ax.hist(truth, weights=np.ones(n_y)/float(n_y),
#             bins=np.arange(min(truth), max(truth) + binwidth, binwidth),
#             color='orange', alpha=0.5,
#             label='True distribution')
#     ax.legend()
# # Add title
# # g.fig.subplots_adjust(top=0.8)  # adjust the Figure in rp
# # g.fig.suptitle("Unidentifiable, sigma_v=0.5")
# # Change x axis label
# g.set_axis_labels("Magnitude of bias", "Probability")
# # Save plot to pdf
# # plt.savefig("ppredictive-groupmean-30nco.pdf", format="pdf", bbox_inches="tight")
# plt.show()
