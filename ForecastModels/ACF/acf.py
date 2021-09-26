import numpy as np
import pandas as pd
import scipy.stats as st


def compute_sample_correlation(ts_1, ts_2, var=None, mean=None, T=None):
    T = len(ts_1) if T is None else T
    if mean is None:
        mean_s1, mean_s2 = 0, 0
        for i in range(len(ts_1)):
            mean_s1 += ts_1[i]
            mean_s2 += ts_2[i]
        mean_s1, mean_s2 = mean_s1 / T, mean_s2 / T
    else:
        mean_s1, mean_s2 = mean, mean

    cov, var_s1, var_s2 = 0, 0, 0
    for i in range(len(ts_1)):
        cov += (ts_1[i] - mean_s1) * (ts_2[i] - mean_s2)
        var_s1 += (ts_1[i] - mean_s1) ** 2
        var_s2 += (ts_2[i] - mean_s2) ** 2

    if var is None:
        cov = cov / (T - 1)
        var_s1 = var_s1 / (T - 1)
        var_s2 = var_s2 / (T - 1)
        sc = cov / (var_s1 * var_s2) ** 0.5
    else:
        cov = cov / (T - 1)
        sc = cov / var  # *(T - 1)/T
    return sc[0]


def compute_acf(ts, nlags=10, alpha=5. / 100):
    # ts -> expected as dataFrame

    T = len(ts)

    # ACF estimates ---------------------------------------------------------
    mean = ts.mean().values[0]
    variance = ts.var(ddof=1).values[0]
    acf = [compute_sample_correlation(ts[lag:].values,
                                      ts[0:T - lag].values,
                                      var=variance,
                                      mean=mean,
                                      T=len(ts)
                                      ) for lag in range(nlags + 1)]
    acf = np.array(acf)
    # -----------------------------------------------------------------------

    # t-Examples_and_Tests null hyphotesis estimates -------------------------------------
    dist = st.norm()  # st.t(df=T - 1)  # st.t(df=T - nlags)  # st.t(df=T - 1), st.norm()
    crit_l = dist.ppf(q=alpha / 2)
    crit_u = dist.ppf(q=1 - alpha / 2)

    t_ratio, rho_crit, p_values, t_crit, null_reject = [], [], [], [], []

    for lag in range(0, nlags + 1):
        std = ((1 + 2 * acf[1:lag].sum()) / T) ** 0.5
        t_val = acf[lag] / std
        t_ratio.append(t_val)
        ci_l = crit_l * std  # t_crit_l*std + acf[lag]
        ci_u = crit_u * std  # t_crit_u * std + acf[lag]
        rho_crit.append([ci_l, ci_u])
        t_crit.append(crit_u)
        # p_values.append(2*(1 - dist.cdf(np.abs(t_ratio[-1]))))
        p_values.append(2 * (1 - dist.cdf(np.abs(t_val))))
        null_reject.append(abs(t_val) > crit_u)

    t_ratio[0], rho_crit[0], p_values[0], t_crit[0], null_reject[0] = None, None, None, None, None
    # -----------------------------------------------------------------------

    # Portmanteau, (Ljung and Box modification) Hythotesis Examples_and_Tests estimates --
    q_stats, p_value_q, null_reject_q = [], [], []
    for lag in range(0, nlags + 1):
        chi_dist = st.chi2(df=lag)
        sumation = 0
        for i in range(1, lag + 1):
            sumation += acf[i] ** 2 / (T - i)

        q = T * (T + 2) * sumation
        q_stats.append(q)
        pq = 1 - chi_dist.cdf(q)
        p_value_q.append(pq)
        null_reject_q.append(pq < alpha)

    q_stats[0], p_value_q[0], null_reject_q[0] = None, None, None
    # -----------------------------------------------------------------------

    # Tests summary var -----------------------------------------------------
    adf_test_summ_dict = dict()
    adf_test_summ_dict['t_test_any_null_reject'] = np.any(null_reject[1:])
    adf_test_summ_dict['t_test_all_null_reject'] = np.all(null_reject[1:])
    adf_test_summ_dict['t_test_lags_reject'] = np.arange(1, nlags + 1)[null_reject[1:]]

    adf_test_summ_dict['q_test_any_null_reject'] = np.any(null_reject_q[1:])
    adf_test_summ_dict['q_test_all_null_reject'] = np.all(null_reject_q[1:])
    adf_test_summ_dict['q_test_lags_reject'] = np.arange(1, nlags + 1)[null_reject_q[1:]]

    # m_bpf = int(np.log(T))
    # adf_test_summ_dict['m_bpf'] = m_bpf
    # adf_test_summ_dict['p_value_bpf'] = p_value_q[m_bpf]
    # adf_test_summ_dict['null_reject_bpf'] = null_reject_q[m_bpf]
    # -----------------------------------------------------------------------

    return pd.DataFrame(dict(lag=np.arange(nlags + 1), acf=acf, null_crit_I=rho_crit, t_ratio=t_ratio,  # t_crit=t_crit,
                             p_values=p_values, null_reject=null_reject,
                             q_stats=q_stats, p_value_q=p_value_q, null_reject_q=null_reject_q)), adf_test_summ_dict
