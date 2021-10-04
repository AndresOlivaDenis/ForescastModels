import os

from SimulationScenarioEngine.DataPreprocesor.dataPreprocess import DataPreprocess

# Data Preprocess =====================================================================================================
from ForecastModels.OptionPricingForescast.OPFUtils import build_return_generator_by_hist_data_sample

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

rghd = build_return_generator_by_hist_data_sample(data_df)
print(rghd)

# rghd.compute_price_path_moments_from_sample(s0=2500, sample_size=10000)
# rghd.plot_yearly_price_path_sims(2500, n=10)
# rghd.plot_samples(year_cum_returns)


begin_date = "2021-10-01"
end_date = "2021-11-01"

daily_price_paths_bt_dates = rghd.simulate_daily_price_paths_between_dates(s0=2500,
                                                                           begin_date=begin_date,
                                                                           end_date=end_date,
                                                                           reshape=True)

daily_price_paths_bt_dates_series = rghd.simulate_daily_price_paths_between_dates_series(s0=2500,
                                                                                         begin_date=begin_date,
                                                                                         end_date=end_date)
