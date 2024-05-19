import numpy as np
import pytensor.tensor as pt
import pymc as pm
print(f"Running on PyMC v{pm.__version__}")


# NOTE: Use only for binary treatment contrast, a.k.a. K = 1
# ate: I \times K matrix of ATE estimates
# se: I \times K matrix of s.e. of the ATE estimates

# NOTE: Probably will not work. "Spike n Slap GP" usually put selection prior on the length-scale.


def run_hetero_gp(ate, se, x,
                  sample, tune, chain,
                  phi_d=None, phi_m=None,
                  alpha_d=1, beta_d=1, alpha_m=1, beta_m=1,
                  x_star=None,
                  random_seed=123,
                  target_accept=0.8, cores=1,
                  sigma_v=1
                  ):
    # Set the number of NCOs
    I = ate.shape[0]

    # Set the number of scenarios/treatment contrasts
    K = ate.shape[1]

    # Declare the dimensions of the data and parameters.
    # This helps with organizing the indices of the posterior samples
    coords_nco = ['nco_'+str(i) for i in range(I)]
    coords_setting = ['setting_'+str(i) for i in range(K)]
    coords = {'nco': coords_nco, 'setting': coords_setting}

    # Initialize a model
    model = pm.Model(coords=coords)
    with model:

        # State input data as shared variables. Not sure if this is neccesary, but seems to be recommended in the documentation
        Y_t = pm.MutableData('Y', ate, dims=('nco', 'setting'))
        Sigma_t = pm.MutableData('Sigma', se, dims=('nco', 'setting'))
        # currently a vector of outcome probabilities
        x_t = pm.MutableData('x', x, dims=('nco', 'probability'))

        # Priors for unknown model parameters
        # phi_d: probability of having zero bias
        if phi_d is None:
            phi_d = pm.Beta('phi_d', alpha=alpha_d, beta=beta_d)
        # phi_m: probability of having a positive bias
        if phi_m is None:
            phi_m = pm.Beta('phi_m', alpha=alpha_m, beta=beta_m)
        # For one scenario: model the mean function as GP with square exponential kernel
        # ls: lengthscale controls wiggliness of mean function, between 0.0 and 0.9 with 95% since x \in (0, 1)
        ls = 0.5  # pm.InverseGamma('ls', alpha=3.35, beta=0.9)
        # sigma: controls magnitude of mean function
        # NOTE: set hyper-param so that amplitude of mean function is <= max(abs(ate)) w.p. 0.95
        # pm.HalfNormal('sigma', sigma=np.max(np.abs(ate))/3)
        sigma = pm.HalfNormal('sigma', sigma=10)  # np.max(np.abs(ate))/3
        # define a GP prior and its covariance
        gp_m = pm.gp.Latent(
            # mean_func=pm.gp.mean.Constant(np.mean(np.abs(ate))),
            cov_func=sigma**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=ls))
        # assign the GP prior to observations
        m = gp_m.prior('m', x_t, dims='nco', jitter=1e-6)

        # For one scenario: model the variance as a GP with square exponential kernel
        # ls: lengthscale controls wiggliness of variance function
        # ls_v = pm.InverseGamma('ls_v', alpha=3.35, beta=0.9)
        # sigma: controls magnitude of variance function
        # NOTE: set hyper-param so P[log_v <= np.log(np.std(ate))] \approx 0.99
        # so that the amplitude of the std from m(x) <= std(data) w.p. 99%
        # sigma_v = pm.HalfNormal('sigma_v', sigma=10)
        v = pm.HalfNormal('v', sigma=np.max(np.abs(ate))/3)
        # # sigma_v = 1
        # # define a GP prior and its covariance
        # gp_v = pm.gp.Latent(
        #     mean_func=pm.gp.mean.Constant(
        #         np.log(np.std(np.abs(ate)))),
        #     cov_func=sigma_v**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=ls))
        # # assign the GP prior to observations
        # log_v = gp_v.prior('log_v', x_t, dims='nco', jitter=1e-6)
        # v = pm.Deterministic("v", pm.math.exp(log_v))

        # sample indicator of a zero bias
        delta = pm.Bernoulli('delta', p=phi_d, size=I, dims='nco')
        # sample indicator of a positive bias
        M = pm.Bernoulli('M', p=phi_m, size=I, dims='nco')
#         M_iden = pm.Deterministic('M_iden', M * pm.math.sgn(M[I-1])) # enforce the first M to be positive for identifiability
        # composite of m and v
        f = pm.Normal('f', mu=m, sigma=v, dims='nco')
        # Important: all scenarios will have one mu(x), aka the same mean
        mu = pm.Deterministic('mu',
                              pt.outer((1-delta)*(2*M-1)*f, pt.ones(K)),
                              dims=('nco', 'setting'))

        # Likelihood model
        # TODO: make this a truncated likelihood -2 to 2?
        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=Sigma_t,
                          observed=Y_t, dims=('nco', 'setting'))

        # Sample posteriors
        trace = pm.sample(sample, tune=tune, chains=chain,
                          target_accept=target_accept, random_seed=random_seed, cores=cores)

        # Sample posterior predictive
        if x_star is not None:
            m_star = gp_m.conditional('m_star', x_star, jitter=1e-6)
            # log_v_star = gp_v.conditional('log_v_star', x_star, jitter=1e-6)
            # v_star = pm.Deterministic('v_star', pm.math.exp(log_v_star))
            # f_star = pm.Normal('f_star', mu=m_star, sigma=v_star)
            f_star = pm.Normal('f_star', mu=m_star, sigma=v)
            delta_star = pm.Bernoulli('delta_star', p=phi_d, size=len(x_star))
            mu_star = pm.Deterministic('mu_star', f_star*(1-delta_star))
            ppc = pm.sample_posterior_predictive(trace, var_names=['m_star',
                                                                   #    'log_v_star', 'v_star',
                                                                   'f_star',
                                                                   'delta_star', 'mu_star'])
            trace.extend(ppc)

    return (model, trace)


# NOTE: Use only for binary treatment contrast, a.k.a. K = 1
# Remove constraints that beta is non-negative so the sampler explores the space
# more efficiently and removes divergences in Hamiltonian MC
# Post-process:
#      m_ik = ((2m_i-1)*sign(beta_k) + 1) / 2
#      beta_k = abs(beta_k)
# ate: I \times K matrix of ATE estimates
# se: I \times K matrix of s.e. of the ATE estimates
# phi_d: probability of the bias being zero, if None places hierarchical priors on them
# phi_m: probability of the bias being positive, if None places hierarchical priors on them
def get_mean_model(ate, se, phi_d=None, phi_m=None,
                   nu=None, tau=None,
                   alpha_d=1, beta_d=1, alpha_m=1, beta_m=1
                   ):
    # Set the number of NCOs
    I = ate.shape[0]

    # Set the number of scenarios/treatment contrasts
    K = ate.shape[1]

    # Declare the dimensions of the data and parameters.
    # This helps with organizing the indices of the posterior samples
    coords_nco = ['nco_'+str(i) for i in range(I)]
    coords_setting = ['setting_'+str(i) for i in range(K)]
    coords = {'nco': coords_nco, 'setting': coords_setting}

    # Initialize a model
    model = pm.Model(coords=coords)
    with model:

        # State input data as shared variables. Not sure if this is neccesary, but seems to be recommended in the documentation
        Y_t = pm.MutableData('Y', ate, dims=('nco', 'setting'))
        # TODO: update to include covariances
        Sigma_t = pm.MutableData('Sigma', se, dims=('nco', 'setting'))

        # Priors for unknown model parameters
        if nu is None:
            nu_t = pm.Normal('nu', mu=0, sigma=100)
        else:
            nu_t = nu
        if tau is None:
            tau_t = pm.InverseGamma('tau', 0.001, 0.001)
        else:
            tau_t = tau
        if phi_d is None:
            phi_d_t = pm.Beta('phi_d', alpha=alpha_d, beta=beta_d)
        else:
            phi_d_t = phi_d
        if phi_m is None:
            phi_m_t = pm.Beta('phi_m', alpha=alpha_m, beta=beta_m)
        else:
            phi_m_t = phi_m
        beta_t = pm.Normal(
            'beta',
            mu=nu_t,
            sigma=pt.sqrt(tau_t),
            size=K,
            dims='setting')
        # sigma_t = pm.InverseGamma('sigma', alpha=alpha_sigma, beta=beta_sigma, size=K, dims='setting')
        sigma_t = pm.HalfCauchy('sigma', beta=25, size=K, dims='setting')
        delta_t = pm.Bernoulli('delta', p=phi_d_t, size=I, dims='nco')
        m_t = pm.Bernoulli('m', p=phi_m_t, size=I, dims='nco')
        # This seems to work better than the matrix normal prior
        slap_t = pm.Normal('slap', mu=pt.outer(
            (2*m_t-1), beta_t), sigma=pt.outer(pt.ones(I), pt.sqrt(sigma_t)), dims=('nco', 'setting'))
        mu_t = pm.Deterministic('mu', pt.outer(
            (1-delta_t), pt.ones(K))*slap_t, dims=('nco', 'setting'))

        # Likelihood model
        Y_obs = pm.Normal('Y_obs', mu=mu_t, sigma=Sigma_t,
                          observed=Y_t, dims=('nco', 'setting'))

    return model

# x: an I-vector of outcome probabilities


def run_polynomial(ate, se, x, sample, tune, chain,
                   alpha_d=1, beta_d=1, alpha_m=1, beta_m=1,
                   x_star=None, phi_d=None, phi_m=None,
                   nu=None, tau=None,
                   random_seed=123, target_accept=0.8, cores=1):

    # Set the number of NCOs
    I = ate.shape[0]

    # Set the number of scenarios/treatment contrasts
    K = ate.shape[1]

    # Declare the dimensions of the data and parameters.
    # This helps with organizing the indices of the posterior samples
    coords_nco = ['nco_'+str(i) for i in range(I)]
    coords_setting = ['setting_'+str(i) for i in range(K)]
    coords = {'nco': coords_nco,
              'setting': coords_setting,
              'poly': ['intercept', 'x', 'x2', 'x3']}

    # Initialize a model
    model = pm.Model(coords=coords)
    with model:

        # State input data as shared variables. Not sure if this is neccesary, but seems to be recommended in the documentation
        Y_t = pm.MutableData('Y', np.squeeze(ate), dims='nco')
        # TODO: update to include covariances
        Sigma_t = pm.MutableData('Sigma', np.squeeze(se), dims='nco')
        # polynomials of the outcome probabilities (I x 4)
        x_t = np.squeeze(np.stack([np.ones_like(x), x, x**2, x**3], axis=-1))

        # Priors for unknown model parameters
        if nu is None:
            nu_t = pm.Normal('nu', mu=0, sigma=100)
        else:
            nu_t = nu
        if tau is None:
            tau_t = pm.HalfCauchy('tau', 25)
        else:
            tau_t = tau
        if phi_d is None:
            phi_d_t = pm.Beta('phi_d', alpha=alpha_d, beta=beta_d)
        else:
            phi_d_t = phi_d
        if phi_m is None:
            phi_m_t = pm.Beta('phi_m', alpha=alpha_m, beta=beta_m)
        else:
            phi_m_t = phi_m
        beta_t = pm.Normal(
            'beta',
            mu=nu_t,
            sigma=pt.sqrt(tau_t),
            size=4,
            dims='poly')
        sigma_t = pm.HalfCauchy('sigma', beta=25, size=1)
        delta_t = pm.Bernoulli('delta', p=phi_d_t, size=I, dims='nco')
        m_t = pm.Bernoulli('m', p=phi_m_t, size=I, dims='nco')
        # This seems to work better than the matrix normal prior
        slap_t = pm.Normal(
            'slap',
            mu=(2*m_t-1)*pt.dot(x_t, beta_t),
            sigma=pt.sqrt(sigma_t),
            dims='nco')
        mu_t = pm.Deterministic(
            'mu',
            (1-delta_t)*slap_t,
            dims='nco')

        # Likelihood model
        print("Y_obs")
        Y_obs = pm.Normal('Y_obs', mu=mu_t, sigma=Sigma_t,
                          observed=Y_t, dims='nco')

        # Sample posteriors
        print("trace")
        trace = pm.sample(sample, tune=tune, chains=chain,
                          target_accept=target_accept, random_seed=random_seed, cores=cores)

        # Sample posterior predictive
        if x_star is not None:
            I_star = len(x_star)
            x_star_t = np.squeeze(np.stack(
                [np.ones_like(x_star), x_star, x_star**2, x_star**3], axis=-1))
            print(x_star_t.shape)
            print("delta_star")
            delta_star = pm.Bernoulli('delta_star', p=phi_d_t, size=I_star)
            print("slap_star")
            slap_star = pm.Normal(
                'slap_star',
                # assuming m_star are all 1
                mu=pt.dot(x_star_t, beta_t),
                sigma=pt.sqrt(sigma_t),
                size=I_star)
            print("mu_star")
            mu_star = pm.Deterministic(
                'mu_star', pt.abs((1-delta_star)*slap_star))
            print("ppc")
            ppc = pm.sample_posterior_predictive(
                trace,
                var_names=['delta_star', 'slap_star', 'mu_star'])
            print("extend trace")
            trace.extend(ppc)

    return (model, trace)
