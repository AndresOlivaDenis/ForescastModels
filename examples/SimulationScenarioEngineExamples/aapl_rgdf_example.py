import os

from SimulationScenarioEngine.DataPreprocesor.dataPreprocess import DataPreprocess

# Data Preprocess =====================================================================================================
from ForecastModels.OptionPricingForescast.OPFUtils import build_return_generator_by_distribution_fit

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
rgdf = build_return_generator_by_distribution_fit(data_df=data_df)
rgdf.plot_fitting_results(n=2, bins=50)
print(rgdf)
# =====================================================================================================================


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


