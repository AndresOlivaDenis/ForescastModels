import os
import pandas as pd

from SimulationScenarioEngine.DataPreprocesor.dataPreprocess import DataPreprocess

from ForecastModels.OptionPricingForescast.OPFUtils import build_return_generator_by_distribution_fit, \
    build_return_generator_by_hist_data_sample
from ForecastModels.OptionPricingForescast.OptionsForecastModel import OptionsForecastModels
from ForecastModels.OptionPricingForescast.OptionsImpliedForecastModels import OptionsImpliedForecastModels
from ForecastModels.OptionPricingForescast.utils import OFMOptimizationApproach

if __name__ == '__main__':
    # Loadding of Data ===================================================================================================
    folder_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))) + '/data/YF'
    file_name = "AAPL.csv"
    load_file_args = dict(ref_colum_date="Date", ref_colum_price="Adj Close", folder_path=folder_path,
                          date_threshold=pd.datetime(2020, 3, 1),
                          )

    data_df = DataPreprocess.load_file_content(file_name=file_name, load_file_args=load_file_args)
    # ====================================================================================================================

    # ----------------------------------------------
    S0_ = 140.25
    # K_, C_price, P_price = 135, 8.43, 2.98
    K_, C_price, P_price = 141, 4.65, 5.25
    # ----------------------------------------------

    # Option implied forecast model by samplig ------------------------------------------------------

    ofm_s = OptionsForecastModels("Normal", method=OFMOptimizationApproach.SAMPLING, sampling_size=10000)
    ofm_s.fit_distribution_to_option_prices(call_price=C_price, put_price=P_price, S0=S0_, K=K_,
                                          current_date=None, expiration_date=None)
    distribution_parameters_s = ofm_s.optimize_distribution_parameters_dict.copy()
    print("distribution_parameters_s:", distribution_parameters_s)

    ofm_i = OptionsForecastModels("Normal", method=OFMOptimizationApproach.INTEGRATING)
    ofm_i.fit_distribution_to_option_prices(call_price=C_price, put_price=P_price, S0=S0_, K=K_,
                                            current_date=None, expiration_date=None)
    distribution_parameters_i = ofm_i.optimize_distribution_parameters_dict.copy()
    print("distribution_parameters_i:", distribution_parameters_i)

    K_list = [131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150]
    call_prices_list = [11.5, 10.7, 9.9, 9.15, 8.43, 7.7, 7.05, 6.4, 5.75, 5.2, 4.65, 4.13, 3.65, 3.2, 2.82, 2.45, 2.13,
                        1.85, 1.59, 1.37]
    put_prices_list = [2.05, 2.24, 2.46, 2.72, 2.98, 3.28, 3.6, 3.95, 4.35, 4.8, 5.25, 5.73, 6.25, 6.85, 7.45, 8.1,
                       8.77, 9.5, 10.25, 11.05]
    comparison_df = pd.DataFrame(index=K_list)
    comparison_df["C"] = call_prices_list
    comparison_df["P"] = put_prices_list

    call_estimate_opt_s, put_estimate_opt_s = ofm_s.price_vainilla_options(K_list, S0_, as_series=True)
    call_estimate_opt_i, put_estimate_opt_i = ofm_i.price_vainilla_options(K_list, S0_, as_series=True)

    comparison_df["C_s"] = call_estimate_opt_s.copy()
    comparison_df["P_s"] = put_estimate_opt_s.copy()
    comparison_df["C_i"] = call_estimate_opt_i.copy()
    comparison_df["P_i"] = put_estimate_opt_i.copy()


    print("C_price, P_price:", (C_price, P_price) )
    print("C_s, P_s: ", (comparison_df["C_s"][K_], comparison_df["P_s"][K_]))
    print("C_i, P_i: ", (comparison_df["C_i"][K_], comparison_df["P_i"][K_]))
    print("call_estimate_opt_i", call_estimate_opt_i)
    print("put_estimate_opt_i", put_estimate_opt_i)

    print(comparison_df)

