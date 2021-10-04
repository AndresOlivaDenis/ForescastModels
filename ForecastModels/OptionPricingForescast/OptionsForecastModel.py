import os
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import scipy.optimize as opt
import scipy.integrate as integrate
from scipy.integrate import IntegrationWarning

from ForecastModels.OptionPricingForescast.utils import DistributionCatalogE, OFMOptimizationApproach


class OptionsForecastModels(object):
    def __init__(self, distribution_name, fixed_params_dict=None, method=OFMOptimizationApproach.INTEGRATING,
                 sampling_size=10000):
        self.distribution_name = distribution_name
        self.fixed_params_dict = fixed_params_dict  # TODO.
        self.distribution_obj_ref = DistributionCatalogE(distribution_name).distribution_obj

        self.method = method
        self.sampling_size = sampling_size
        self.optimize_distribution_parameters_dict = None

        self.option_characteristics = None

    def set_optimize_distribution_parameters_dict(self, distribution_parameters_dict):
        self.optimize_distribution_parameters_dict = distribution_parameters_dict.copy()

    def fit_distribution_to_option_prices(self, call_price, put_price, S0, K, current_date=None, expiration_date=None,
                                          ):
        self.option_characteristics = dict()
        self.option_characteristics["call_price"] = call_price
        self.option_characteristics["put_price"] = put_price
        self.option_characteristics["S0"] = K
        self.option_characteristics["current_date"] = current_date
        self.option_characteristics["expiration_date"] = expiration_date

        distribution_parameters_values = self._fit_distribution_to_option_prices(call_price=call_price,
                                                                                 put_price=put_price,
                                                                                 S0=S0,
                                                                                 K=K)
        self.optimize_distribution_parameters_dict = distribution_parameters_values.copy()
        return distribution_parameters_values.copy()

    def _fit_distribution_to_option_prices(self, call_price, put_price, K, S0):

        if self.method is OFMOptimizationApproach.SAMPLING:
            cost_function = OptionsForecastModels.cost_function_sample
        else:
            cost_function = OptionsForecastModels.cost_function_integrate

        x0 = self.fit_start()
        function = lambda x: cost_function(call_price=call_price,
                                           put_price=put_price,
                                           K=K,
                                           S0=S0,
                                           size=self.sampling_size,
                                           distribution=self.distribution_obj_ref,
                                           x=x)

        param_list = opt.fmin(function, x0=x0, disp=False, maxiter=300, xtol=1e-12)
        return OptionsForecastModels.map_distribution_parameters_values(self.distribution_name,
                                                                        distribution_parameters_values=param_list)

    def price_vainilla_options(self, K_list, S0, as_series=True):
        call_prices, put_prices = [], []
        distribution_params = list(self.optimize_distribution_parameters_dict.values())

        if self.method is OFMOptimizationApproach.SAMPLING:
            price_method = OptionsForecastModels.price_vanilla_options_by_sampling
        else:
            price_method = OptionsForecastModels.price_vanilla_options_by_integrating

        for K in K_list:
            call_estimate, put_estimate = price_method(K=K,
                                                       S0=S0,
                                                       size=self.sampling_size,
                                                       distribution=self.distribution_obj_ref,
                                                       distribution_parameters=distribution_params)
            call_prices.append(call_estimate)
            put_prices.append(put_estimate)

        if as_series:
            call_prices = pd.Series(data=call_prices, index=K_list)
            put_prices = pd.Series(data=put_prices, index=K_list)

        return call_prices, put_prices

    def fit_start(self):
        return list(DistributionCatalogE(self.distribution_name).parameters_dict.values())

    @staticmethod
    def cost_function_sample(call_price, put_price, K, S0, size, distribution, x):
        try:
            call_estimate, put_estimate = OptionsForecastModels.price_vanilla_options_by_sampling(K=K,
                                                                                                  S0=S0,
                                                                                                  size=size,
                                                                                                  distribution=distribution,
                                                                                                  distribution_parameters=x)
            error = OptionsForecastModels.error_function(call_estimate=call_estimate, put_estimate=put_estimate,
                                                         call_price=call_price, put_price=put_price)
            return error
        except Exception as e:
            return np.nan

    @staticmethod
    def cost_function_integrate(call_price, put_price, K, S0, size, distribution, x):
        try:
            call_estimate, put_estimate = OptionsForecastModels.price_vanilla_options_by_integrating(K=K,
                                                                                                     S0=S0,
                                                                                                     size=size,
                                                                                                     distribution=distribution,
                                                                                                     distribution_parameters=x)
            error = OptionsForecastModels.error_function(call_estimate=call_estimate, put_estimate=put_estimate,
                                                         call_price=call_price, put_price=put_price)
            return error
        except Exception as e:
            return np.nan

    @staticmethod
    def error_function(call_estimate, put_estimate, call_price, put_price):
        call_estimate_diff = np.abs(call_estimate - call_price)
        put_estimate_diff = np.abs(put_estimate - put_price)
        error = (call_estimate_diff / call_price) ** 2 + (put_estimate_diff / put_price) ** 2
        # call_estimate_log_diff = np.log(call_price_estimate) - np.log(call_price)
        # put_estimate_log_diff = np.log(put_price_estimate) - np.log(put_price)
        # error = call_estimate_log_diff ** 2 + put_estimate_log_diff ** 2
        return error

    @staticmethod
    def map_distribution_parameters_values(distribution_name, distribution_parameters_values):
        parameters_names = DistributionCatalogE(distribution_name).parameters_dict.keys()
        distribution_parameters_dict = dict()
        for x, y in zip(parameters_names, distribution_parameters_values):
            distribution_parameters_dict[x] = y
        return distribution_parameters_dict

    @staticmethod
    def price_vanilla_options_by_sampling(K, S0, size, distribution, distribution_parameters):
        simulated_final_prices = OptionsForecastModels.generate_final_prices(S0=S0,
                                                                             size=size,
                                                                             distribution=distribution,
                                                                             distribution_parameters=distribution_parameters)
        call = np.maximum(simulated_final_prices - K, 0)
        put = np.maximum(K - simulated_final_prices, 0)
        return np.mean(call), np.mean(put)

    @staticmethod
    def price_vanilla_options_by_integrating(K, S0, size, distribution, distribution_parameters):
        warnings.filterwarnings("ignore", category=IntegrationWarning)

        limit = 1e-5
        distribution_obj = distribution(*distribution_parameters)
        a = distribution_obj.ppf(limit)
        b = distribution_obj.ppf(1 - limit)
        call = integrate.quad(lambda x: np.maximum((S0 * np.exp(x) - K), 0) * distribution_obj.pdf(x), a, b,
                              limit=100, limlst=100, maxp1=100)
        put = integrate.quad(lambda x: np.maximum((K - S0 * np.exp(x)), 0) * distribution_obj.pdf(x), a, b,
                             limit=100, limlst=100, maxp1=100)
        return call[0], put[0]

    @staticmethod
    def generate_final_prices(S0, size, distribution, distribution_parameters):
        returns_samples = distribution(*distribution_parameters).rvs(size)
        return S0 * np.exp(returns_samples)

# TODO:
#   Tests parameter optimization sensibility for differents Strikes (optimize K for K_list) ......
#       test_df:
#           loc, scale, C (objective), P (objective), C_s (estimate for objective), P_s (estimate for objective ...)
#   Include an optimization that minimize K_list (although think what approach is more worth!) (maybe mean of, bool var -> all agree, ect ...)
#   See if maybe T distribution can handle "same" parameters for all K

# TODO NEXT:
#   Extend class as "a price model": (maybe new child class!)
#   Create functions to eval prices (forecast functions) (include annualized conversions)
#   Create functions to print returns estimates (include annualized conversions)
#   Include probabilities


# Forecast "strategies":
#   mean of estimates
#   Fit for all K (prev tests for how far away is from single and mean estimates)
#   If all agree & mean of estimates
#   Prob estimates
#   Use ask bid Option prices ? (if with both sig of estimates agree -> more prob of neg!)

# Forecast:
#   Maybe for testing: Try to create a general DF that them with group by & queries allow to tests forecast strategies
#       Loop over historical data and fit for C P (& retrieve future value in expiration date)

#       To Include in:
#           Forecast (of final date) & CI & Prob (using midvalues, ask and bid)
#           Annualized forecast (this migth be nice to check differents horizonts expectations)
#           Include C estim & P estim (to double check!)
#           Real price in expiration_date


# ie. strategy eval using df:
#   Group by expiration_date
#   Mean, of if all agree -> ...

# ie. strategy eval using df:
#   (eval with horizon is better):
#       Filter ....

