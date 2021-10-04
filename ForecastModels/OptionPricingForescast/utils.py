from enum import Enum
import scipy.stats as st
import numpy as np
import sys


class OFMOptimizationApproach(Enum):
    def __new__(cls, name):
        object_new = object.__new__(cls)
        object_new._value_ = name
        return object_new

    SAMPLING = ('Sampling', )
    INTEGRATING = ('Integrating', )


class DistributionCatalogE(Enum):

    def __new__(cls, name, distribution_obj, parameters_dict):
        object_new = object.__new__(cls)
        object_new._value_ = name
        object_new.distribution_obj = distribution_obj
        object_new.parameters_dict = parameters_dict
        return object_new

    BETA = ('Beta', st.beta, dict(a=2.0, b=2.0, loc=0.0, scale=1.0))
    CAUCHY = ('Cauchy', st.cauchy, dict(loc=0.0, scale=1.0))
    CHISQUARE = ('Chisquare', st.chi2, dict(df=1, loc=0.0, scale=1.0))
    EXPONENTIAL = ('Exponential', st.expon, dict(loc=0.0, scale=1.0))
    F = ('F', st.f, dict(dfn=1, dfd=1, loc=0.0, scale=1.0))
    GAMMA = ('Gamma', st.gamma, dict(a=1.0, loc=0.0, scale=1.0))
    GUMBEL = ('Gumbel', st.gumbel_r, dict(loc=0.0, scale=1.0))
    LAPLACE = ('Laplace', st.laplace, dict(loc=0.0, scale=1.0))
    LOGISTIC = ('Logistic', st.logistic, dict(loc=0.0, scale=1.0))
    LOGNORMAL = ('Lognormal', st.lognorm, dict(s=1.0, loc=0.0, scale=1.0))
    NORMAL = ('Normal', st.norm, dict(loc=0.0, scale=1.0))
    T = ('T', st.t, dict(df=1, loc=0.0, scale=1.0))
    TRIANGULAR = ('Triangular', st.triang, dict(c=0.5, loc=0.0, scale=1.0))
    UNIFORM = ('Uniform', st.uniform, dict(loc=0.0, scale=1.0))
    VOMISES = ('Vonmises', st.vonmises, dict(kappa=1.0, loc=0.0, scale=1.))
    WALD = ('Wald', st.wald, dict(loc=0.0, scale=1.0))
    WEIBULL = ('Weibull', st.weibull_min, dict(c=3.0, loc=0.0, scale=1.0))
    

def integrate_by_simpson38_(func, a, b, n_div=150):
    x = np.linspace(a, b, n_div + 1)
    x_1 = x[[0, n_div]]
    x_2 = x[np.arange(3, n_div, 3)]
    x_3 = np.delete(x, np.arange(3, n_div, 3))[1:-1]
    integral = func(x_1).sum() + (2 * func(x_2)).sum() + (3 * func(x_3)).sum()
    return integral * 3 * ((b - a) / n_div) / 8


def integrate_by_simpson38_list_(func, a, b, n_div=300):
    x = np.array([np.linspace(i, j, n_div + 1) for i, j in zip(a, b)])
    x_1 = x[:, [0, n_div]]
    x_2 = x[:, [np.arange(3, n_div, 3)]].reshape(len(b), -1)
    x_3 = np.array([np.delete(x[i, :], np.arange(3, n_div, 3))[1:-1] for i in range(len(a))])
    integral = func(x_1).sum(axis=1) + (2 * func(x_2)).sum(axis=1) + (3 * func(x_3)).sum(axis=1)
    return integral * 3 * ((b - a) / n_div) / 8
