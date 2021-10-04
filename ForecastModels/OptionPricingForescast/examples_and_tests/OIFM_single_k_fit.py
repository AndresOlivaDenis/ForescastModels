import os
import pandas as pd


from SimulationScenarioEngine.DataPreprocesor.dataPreprocess import DataPreprocess

from ForecastModels.OptionPricingForescast.OPFUtils import build_return_generator_by_distribution_fit, \
    build_return_generator_by_hist_data_sample
from ForecastModels.OptionPricingForescast.OptionsImpliedForecastModels import OptionsImpliedForecastModels

if __name__ == '__main__':
    # Loadding of Data ===================================================================================================
    folder_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))) + '/data/YF'
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
    # K_, C_price, P_price = 135, 8.43, 2.98
    K_, C_price, P_price = 141, 4.65, 5.25
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

    # expectation_obtained = []
    # for i in range(20):
    #     oifm.optimize_generator_expectation(C_price, P_price,
    #                                         S0=S0_, K=K_,
    #                                         current_date=current_date_, expiration_date=expiration_date_,
    #                                         n=10000)
    #     expectation_obtained.append(oifm.optimize_expectation)

