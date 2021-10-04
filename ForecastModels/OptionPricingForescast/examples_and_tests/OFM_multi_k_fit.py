import os
import pandas as pd

from SimulationScenarioEngine.DataPreprocesor.dataPreprocess import DataPreprocess

from ForecastModels.OptionPricingForescast.OPFUtils import build_return_generator_by_distribution_fit, \
    build_return_generator_by_hist_data_sample
from ForecastModels.OptionPricingForescast.OptionsForecastModel import OptionsForecastModels
from ForecastModels.OptionPricingForescast.OptionsImpliedForecastModels import OptionsImpliedForecastModels
from ForecastModels.OptionPricingForescast.utils import OFMOptimizationApproach

if __name__ == '__main__':
    S0_ = 140.25

    K_list = [131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150]
    call_prices_list = [11.5, 10.7, 9.9, 9.15, 8.43, 7.7, 7.05, 6.4, 5.75, 5.2, 4.65, 4.13, 3.65, 3.2, 2.82, 2.45, 2.13,
                        1.85, 1.59, 1.37]
    put_prices_list = [2.05, 2.24, 2.46, 2.72, 2.98, 3.28, 3.6, 3.95, 4.35, 4.8, 5.25, 5.73, 6.25, 6.85, 7.45, 8.1,
                       8.77, 9.5, 10.25, 11.05]


    def eval_multi_k(K_list, C_list, P_list, S0):
        model_list = []
        distribution_parameters_list = []
        call_estimate, put_estimate = [], []
        for k, c, p in zip(K_list, C_list, P_list):
            ofm_i = OptionsForecastModels("Normal", method=OFMOptimizationApproach.INTEGRATING)
            ofm_i.fit_distribution_to_option_prices(call_price=c, put_price=p, S0=S0, K=k,
                                                    current_date=None, expiration_date=None)
            distribution_parameters_i = ofm_i.optimize_distribution_parameters_dict.copy()
            call_estimate_opt_i, put_estimate_opt_i = ofm_i.price_vainilla_options([k], S0_, as_series=False)

            call_estimate.append(call_estimate_opt_i[0])
            put_estimate.append(put_estimate_opt_i[0])
            distribution_parameters_list.append(distribution_parameters_i.copy())
            model_list.append(ofm_i)

        estudy_df = pd.DataFrame(index=K_list)
        estudy_df["C"] = C_list
        estudy_df["P"] = P_list
        estudy_df["C estim"] = call_estimate
        estudy_df["P estim"] = put_estimate

        parameters_df = pd.DataFrame(distribution_parameters_list, index=K_list)
        return model_list, estudy_df, parameters_df


    model_list, estudy_df, parameters_df = eval_multi_k(K_list=K_list,
                                                        C_list=call_prices_list,
                                                        P_list=put_prices_list,
                                                        S0=S0_)
