import os

from SimulationScenarioEngine.DataPreprocesor.dataPreprocess import DataPreprocess
from SimulationScenarioEngine.ReturnsGenerators.returnsGeneratorDistributionFit import ReturnsGeneratorDistributionFit

# Data Preprocess =====================================================================================================
from ForecastModels.OptionPricingForescast.OptionsImpliedForecastModels import OptionsImpliedForecastModels

folder_path = os.path.dirname(os.path.dirname(os.getcwd())) + '/data/YF'

file_name = "AAPL.csv"
load_file_args = dict(ref_colum_date="Date",
                      ref_colum_price="Adj Close",
                      folder_path=folder_path,
                      # date_threshold=pd.datetime(2010, 3, 1),
                      )

data_df = DataPreprocess.load_file_content(file_name=file_name, load_file_args=load_file_args)
# data_df_dout = DataPreprocess.drop_out_of_date_median_values(data_df)
# =====================================================================================================================

# Returns generator by distribution fit ==============================================================================

# distribution fit evaluations:

distribution_fitter_params = dict(cost_function="MLE",
                                  # "MLE" # "MethodOfMoments  # TONOTES: MethodOfMoments seems to fit better (moments atleast)!
                                  distribution_list=["Normal",
                                                     # "T"    # TONOTES Seems to not be so good when looking at moments!
                                                     ],
                                  use_fit_method=True)
returns_generator_params_dict = dict(drop_out_of_date_median_values=False)
rgdf = ReturnsGeneratorDistributionFit(data_df=data_df,
                                       returns_generator_params_dict=returns_generator_params_dict)
rgdf.fit_distribution_to_data(distribution_fitter_params=distribution_fitter_params)
# loc_adjustment = 0.1231 / (rgdf.open_days_per_year * rgdf.frames_in_a_day)
# rgdf.adjust_location_for_distribution_obj_fitted(loc_val=0.1231)
# rgdf.adjust_location_for_distribution_obj_fitted(loc_val=0.1231, scale_val=0.25)
# rgdf.plot_fitting_results(n=2, bins=50)
sample_moments_stats, returns_samples = rgdf.compute_moments_from_sample(sample_size=50000)
sample_daily_moments_stats, cum_returns = rgdf.compute_daily_moments_from_sample(daily_sample_size=50000)
sample_yearly_moments_stats, year_cum_returns = rgdf.compute_yearly_moments_from_sample(yearly_sample_size=10000)
print(rgdf)

begin_date = "2021-10-01"
end_date = "2021-11-01"

daily_price_paths_bt_dates = rgdf.simulate_daily_price_paths_between_dates(s0=2500,
                                                                           begin_date=begin_date,
                                                                           end_date=end_date,
                                                                           reshape=True)

daily_price_paths_bt_dates_series = rgdf.simulate_daily_price_paths_between_dates_series(s0=2500,
                                                                                         begin_date=begin_date,
                                                                                         end_date=end_date,
                                                                                         )


options_implied_forecast_model = OptionsImpliedForecastModels()
