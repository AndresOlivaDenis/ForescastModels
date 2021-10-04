import os
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import scipy.optimize as opt

from SimulationScenarioEngine.DataPreprocesor.dataPreprocess import DataPreprocess

from ForecastModels.OptionPricingForescast.OPFUtils import build_return_generator_by_distribution_fit, \
    build_return_generator_by_hist_data_sample


class OptionsImpliedForecastModels(object):
    def __init__(self, return_generator):
        self.rg = return_generator
        self.sample_yearly_moments_stats_old = self.rg.sample_yearly_moments_stats.copy()
        self.optimize_expectation = None
        self.x_opt = None

    def generate_multiple_final_prices(self, S0, current_date, expiration_date, n=10000):
        # simulated_final_prices = []
        # for i in range(n):
        #     price_path_bt_dates = self.rg.simulate_daily_price_paths_between_dates(s0=S0,
        #                                                                            begin_date=current_date,
        #                                                                            end_date=expiration_date,
        #                                                                            reshape=False)
        #     simulated_final_prices.append(price_path_bt_dates[-1])
        # simulated_final_prices = np.array(simulated_final_prices)

        daily_price_paths = self.rg.simulate_multiple_daily_price_paths_between_dates(s0=S0,
                                                                                      begin_date=current_date,
                                                                                      end_date=expiration_date,
                                                                                      size=n)
        simulated_final_prices = daily_price_paths[:, -1]
        return simulated_final_prices

    def price_vanilla_options_mult_K(self, S0, K, current_date, expiration_date, n=10000, as_series=False):
        simulated_final_prices = self.generate_multiple_final_prices(S0=S0,
                                                                     current_date=current_date,
                                                                     expiration_date=expiration_date,
                                                                     n=n)
        call_prices = []
        put_prices = []
        for K_to_eval in K:
            call = np.maximum(simulated_final_prices - K_to_eval, 0)
            put = np.maximum(K_to_eval - simulated_final_prices, 0)

            call_prices.append(np.mean(call))
            put_prices.append(np.mean(put))
        if as_series:
            call_prices = pd.Series(data=call_prices, index=K)
            put_prices = pd.Series(data=put_prices, index=K)

        return call_prices, put_prices

    def price_vanilla_options(self, S0, K, current_date, expiration_date, n=10000):
        simulated_final_prices = self.generate_multiple_final_prices(S0=S0,
                                                                     current_date=current_date,
                                                                     expiration_date=expiration_date,
                                                                     n=n)
        call = np.maximum(simulated_final_prices - K, 0)
        put = np.maximum(K - simulated_final_prices, 0)
        return np.mean(call), np.mean(put)

    def price_vanilla_call_option(self, S0, K, current_date, expiration_date, n=10000):
        simulated_final_prices = self.generate_multiple_final_prices(S0=S0,
                                                                     current_date=current_date,
                                                                     expiration_date=expiration_date,
                                                                     n=n)
        option_values = np.maximum(simulated_final_prices - K, 0)
        return np.mean(option_values)

    def price_vanilla_put_option(self, S0, K, current_date, expiration_date, n=10000):
        simulated_final_prices = self.generate_multiple_final_prices(S0=S0,
                                                                     current_date=current_date,
                                                                     expiration_date=expiration_date,
                                                                     n=n)
        option_values = np.maximum(K - simulated_final_prices, 0)
        return np.mean(option_values)

    def optimize_generator_expectation(self, call_price, put_price, S0, K, current_date, expiration_date, n=10000):
        def expectation_error(x):
            self.rg.adjust_location(x)
            call_price_estimate, put_price_estimate = self.price_vanilla_options(S0=S0, K=K,
                                                                                 current_date=current_date,
                                                                                 expiration_date=expiration_date,
                                                                                 n=n)
            call_estimate_diff = np.abs(call_price_estimate - call_price)
            put_estimate_diff = np.abs(put_price_estimate - put_price)
            error = (call_estimate_diff / call_price) ** 2 + (put_estimate_diff / put_price) ** 2
            # call_estimate_log_diff = np.log(call_price_estimate) - np.log(call_price)
            # put_estimate_log_diff = np.log(put_price_estimate) - np.log(put_price)
            # error = call_estimate_log_diff ** 2 + put_estimate_log_diff ** 2
            return error

        res = opt.minimize_scalar(expectation_error)
        optimize_expectation = res.x
        self.rg.adjust_location(optimize_expectation)
        self.optimize_expectation = optimize_expectation
        return optimize_expectation

    def optimize_generator_expectation_c(self, call_price, put_price, S0, K, current_date, expiration_date, n=10000):
        # TODO: add a const to optimization (sum to call_estimate )( issue -> multivariate optimization)
        pass

    def optimize_generator_expectation_mult_k(self, call_price_series, put_price_series,
                                              S0, current_date, expiration_date, n=10000):
        K = call_price_series.index.to_list()

        def expectation_error(x):
            self.rg.adjust_location(x)
            call_price_estimate, put_price_estimate = self.price_vanilla_options_mult_K(S0=S0,
                                                                                        K=K,
                                                                                        current_date=current_date,
                                                                                        expiration_date=expiration_date,
                                                                                        n=n,
                                                                                        as_series=True)
            call_estimate_diff = (call_price_estimate - call_price_series) / call_price_series
            put_estimate_diff = (put_price_estimate - put_price_series) / put_price_series
            error = (call_estimate_diff ** 2).sum() + (put_estimate_diff ** 2).sum()
            return error

        res = opt.minimize_scalar(expectation_error)
        optimize_expectation = res.x
        self.rg.adjust_location(optimize_expectation)
        self.optimize_expectation = optimize_expectation
        return optimize_expectation













if __name__ == '__main__':
    # Loadding of Data ===================================================================================================
    folder_path = os.path.dirname(os.path.dirname(os.getcwd())) + '/data/YF'
    file_name = "AAPL.csv"
    load_file_args = dict(ref_colum_date="Date", ref_colum_price="Adj Close", folder_path=folder_path,
                          date_threshold=pd.datetime(2020, 3, 1),
                          )

    data_df = DataPreprocess.load_file_content(file_name=file_name, load_file_args=load_file_args)
    # ====================================================================================================================

    rg = build_return_generator_by_distribution_fit(data_df=data_df)
    # rg.adjust_location(loc_val=0.05)

    # rg = build_return_generator_by_hist_data_sample(data_df)
    # rg.adjust_location(0.1)
    #
    print(rg)

    oifm = OptionsImpliedForecastModels(return_generator=rg)

    S0_ = 140.25
    K_ = 135,
    C_price = 8.43
    P_price = 2.98
    current_date_ = "2021-10-01"
    expiration_date_ = "2021-11-05"
    call_option_price = oifm.price_vanilla_call_option(S0=S0_, K=K_,
                                                       current_date=current_date_, expiration_date=expiration_date_)

    put_option_price = oifm.price_vanilla_put_option(S0=S0_, K=K_,
                                                     current_date=current_date_, expiration_date=expiration_date_)

    price_path_bt_dates = rg.simulate_daily_price_paths_between_dates(s0=S0_,
                                                                      begin_date=current_date_,
                                                                      end_date=expiration_date_,
                                                                      reshape=False)

    price_path_bt_dates_series = rg.simulate_daily_price_paths_between_dates_series(s0=S0_,
                                                                                    begin_date=current_date_,
                                                                                    end_date=expiration_date_)

    print("call_option_price:", call_option_price)
    print("put_option_price:", put_option_price)

    oifm.optimize_generator_expectation(C_price, P_price,
                                        S0=S0_, K=K_,
                                        current_date=current_date_, expiration_date=expiration_date_,
                                        n=10000)

    print(oifm.optimize_expectation)

    call_option_price_opt = oifm.price_vanilla_call_option(S0=S0_, K=K_,
                                                           current_date=current_date_, expiration_date=expiration_date_)

    put_option_price_opt = oifm.price_vanilla_put_option(S0=S0_, K=K_,
                                                         current_date=current_date_, expiration_date=expiration_date_)
    print("call_option_price_opt:", call_option_price_opt)
    print("put_option_price_opt:", put_option_price_opt)

    print("C_price:", C_price)
    print("P_price:", P_price)

    K_list = [131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150]
    call_prices, put_prices = oifm.price_vanilla_options_mult_K(S0=S0_, K=K_list,
                                                                current_date=current_date_,
                                                                expiration_date=expiration_date_,
                                                                as_series=True)

    # Todo: tests this! See how mane n is enoght to have "stable" estimates
    # expectation_obtained = []
    # for i in range(20):
    #     oifm.optimize_generator_expectation(C_price, P_price,
    #                                         S0=S0_, K=K_,
    #                                         current_date=current_date_, expiration_date=expiration_date_,
    #                                         n=10000)
    #     expectation_obtained.append(oifm.optimize_expectation)

# TODO:
#   Fit distribution parameter such that .... (maybe include in returns generators)
