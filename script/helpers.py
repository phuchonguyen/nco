import numpy as np

# Define function for statistics: s.e. for the biases
def get_boostrap_rd(indices, ncos, treatment, n):
    b_ncos = ncos[indices, :]
    b_treatment = treatment[indices]
    b_bias = []
    for i in range(n-1):
        for j in range(i+1, n):
            b_bias.append(np.mean(b_ncos[b_treatment==j,:], axis=0) - np.mean(b_ncos[b_treatment==i,:], axis=0))
    return np.array(b_bias).transpose()

def get_boostrap_rr(indices, ncos, treatment, n):
    b_ncos = ncos[indices, :]
    b_treatment = treatment[indices]
    b_bias = []
    for i in range(n-1):
        for j in range(i+1, n):
            b_bias.append(np.mean(b_ncos[b_treatment==j,:], axis=0) / np.mean(b_ncos[b_treatment==i,:], axis=0))
    return np.array(b_bias).transpose()

def cov_exp_quad(x1, x2=None, alpha=1, rho=1):
    if x2 is None:
        x2 = x1
    N1 = x1.shape[0]
    N2 = x2.shape[0]
    K = np.zeros((N1, N2))
    for n1 in range(N1):
        for n2 in range(N2):
            K[n1, n2] = alpha**2 * np.exp(-0.5 * np.linalg.norm(x1-x2)**2 / rho)
    return K

# alpha: sqrt amplitude
# rho: length - scale
# sigma: observation noise variance
def gp_pred_rng(x2, y1, x1, alpha, rho, sigma, delta=1e-6):
    N1 = y1.shape[0]
    N2 = x2.shape[0]
    f2 = np.zeros(N2)
    K =  cov_exp_quad(x1, alpha=alpha, rho=rho) + np.diag(np.repeat(sigma, N1))
    L_K = np.linalg.cholesky(K)
    L_K_div_y1 = np.linalg.inv(np.tril(L_K)) @ y1
    K_div_y1 = (L_K_div_y1.T @ np.linalg.inv(np.tril(L_K))).T
    k_x1_x2 = cov_exp_quad(x1, x2, alpha, rho)
    f2_mu = (k_x1_x2.T @ K_div_y1)
    v_pred = np.linalg.inv(np.tril(L_K)) @ k_x1_x2
    cov_f2 = cov_exp_quad(x2, alpha=alpha, rho=rho) - v_pred.T @ v_pred  + np.diag(np.repeat(delta, N2))
    f2 = np.random.multivariate_normal(f2_mu, cov_f2)
    return f2