from scipy.special import comb, perm
import numpy as np

from autodp import rdp_bank, rdp_acct, dp_acct, privacy_calibrator

from decimal import *

getcontext().prec = 1024


def rdp2dp(rdp, bad_event, alpha):
    """
    convert RDP to DP, ref (Proposition 12):
    Canonne, Clément L., Gautam Kamath, and Thomas Steinke. The discrete gaussian for differential privacy. In NeurIPS, 2020.
    """
    eps_dp = Decimal(rdp) + Decimal(1.0)/Decimal(alpha-1) * ((Decimal(1.0)/Decimal(bad_event)).ln() + Decimal(alpha-1)*Decimal(1-1.0/alpha).ln() - Decimal(alpha).ln())
    return float(eps_dp)


def naive_group_rdp(alpha, sigma, q, m):
    """
    Suppose m=2^c for c \in Z
    """
    alpha = alpha * m

    func_gaussian = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x)

    # declare the moment accountants
    acct1 = rdp_acct.anaRDPacct()

    acct1.compose_poisson_subsampled_mechanisms(func_gaussian, q, coeff=1)

    rdp = acct1.get_rdp([alpha])[0]
    
    c = np.log2(m)
    
    return float(3**c * rdp)


def exact_rdp_upd(alpha, sigma, q, m):
    sum_ = Decimal(0.0)
    
    p_r = [Decimal(comb(m, k, exact=True)) * Decimal(q)**Decimal(k) * Decimal(1-q)**Decimal(m-k) for k in range(0, m+1)]
    
    for i in range(0, m+1):
        sum_ +=  p_r[i]*Decimal(np.e)**(Decimal((alpha-1)*alpha*i**2)/Decimal(2*sigma**2))
    rdp = sum_.ln() / Decimal(alpha-1)
    return float(rdp)


def get_eps_ours(q, sigma, m, iters, bad_event=1e-5):
    opt = 1e10
    alpha_list = list(range(2, 101))
    for alpha in alpha_list:
        opt = min(opt, rdp2dp(iters*exact_rdp_upd(alpha, sigma, q, m), bad_event, alpha))
    return opt


def get_eps_rdp(q, sigma, m, iters, bad_event=1e-5):
    opt = 1e10
    alpha_list = list(range(2, 101))
    for alpha in alpha_list:
        opt = min(opt, rdp2dp(iters*naive_group_rdp(alpha, sigma, q, m), bad_event, alpha))
    return opt


def calibrating_sgm_noise_ours(q, eps, m, delta=1e-5, t=1, err=1e-3):
    """
    Calibrate noise to privacy budgets
    """
    sigma_max = 5000
    sigma_min = 5
    
    def binary_search(left, right):
        mid = (left + right) / 2
        
        lbd = get_eps_ours(q, mid, m, iters=t, bad_event=delta)
        ubd = get_eps_ours(q, left, m, iters=t, bad_event=delta)
        
        if ubd > eps and lbd > eps:    # min noise & mid noise are too small
            left = mid
        elif ubd > eps and lbd < eps:  # mid noise is too large
            right = mid
        else:
            print("an error occurs in binary search")
            return -1
        return left, right
        
    # check
    if get_eps_ours(q, sigma_max, m, iters=t, bad_event=delta) > eps:
        print("noise >", sigma_max)
        return -1
    
    while sigma_max-sigma_min > err:
        sigma_min, sigma_max = binary_search(sigma_min, sigma_max)
    return sigma_max


def calibrating_sgm_noise_rdp(q, eps, m, delta=1e-5, t=1, err=1e-3):
    """
    Calibrate noise to privacy budgets
    """
    sigma_max = 5000
    sigma_min = 5
    
    def binary_search(left, right):
        mid = (left + right) / 2
        
        lbd = get_eps_rdp(q, mid, m, iters=t, bad_event=delta)
        ubd = get_eps_rdp(q, left, m, iters=t, bad_event=delta)
        
        if ubd > eps and lbd > eps:    # min noise & mid noise are too small
            left = mid
        elif ubd > eps and lbd < eps:  # mid noise is too large
            right = mid
        else:
            print("an error occurs in binary search")
            return -1
        return left, right
        
    # check
    if get_eps_rdp(q, sigma_max, m, iters=t, bad_event=delta) > eps:
        print("noise >", sigma_max)
        return -1
    
    while sigma_max-sigma_min > err:
        sigma_min, sigma_max = binary_search(sigma_min, sigma_max)
    return sigma_max

