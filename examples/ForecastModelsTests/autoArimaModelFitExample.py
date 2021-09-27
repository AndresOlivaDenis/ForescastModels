import os
import pandas as pd

from ForecastModels.ModelOne.ARIMAmodel import ARIMAmodel
from ForecastModels.ModelOne.AutoARIMAmodel import AutoARIMAmodel
from ForecastModels.ModelOne.StationarityTests import StationarityTests
from ForecastModels.ModelOne.ARIMASelCriteriaEnum import ARIMASelCriteriaEnum


# Data Loading ========================================================================================================

# path_default = os.path.dirname(os.getcwd()) + '/data'
path_default = os.path.dirname(os.path.dirname(os.getcwd())) + '/data'
path_module3_data = path_default + "/IMFx/Module3_data/"
file_name = "module3_data_PE_Ratios.csv"

module3_data_PE_Rations_df = pd.read_csv(path_module3_data + file_name)
module3_data_PE_Rations_df = module3_data_PE_Rations_df.set_index(['dateid'])
pe_saf_df = module3_data_PE_Rations_df['pe_saf'].dropna()

time_series_df = pe_saf_df.copy()
# =====================================================================================================================

# Model identification ===============================================================================================
time_series_df.plot()

## Detecting stationarity:
stationarity_test_reer = StationarityTests(time_series_df)
stationarity_test_reer.validate_stationarity()
#
## TODO_: Detecting seasonality & Differencing to achieve stationarity
#

## Identify p and q: Autocorrelation and partial autocorrelation plots
ARIMAmodel.plot_correlogram(time_series_df)
correlogram_df = ARIMAmodel.get_correlogram_df(time_series_df)
print(correlogram_df)
# =====================================================================================================================

# Model Estimation ====================================================================================================
auto_ARIMA_model = AutoARIMAmodel(data_df=time_series_df, selection_criteria=ARIMASelCriteriaEnum.R_squared_adj_spc)

# Models summary
print(auto_ARIMA_model.models_summary_df)

# Best model summary and coefficients summary
auto_ARIMA_model.best_model.print_arma_res_summary()


# =====================================================================================================================

# Best Model diagnostics ===============================================================================================
print(auto_ARIMA_model.best_model.residual_checks)
auto_ARIMA_model.best_model.plot_residuals_dignostic()

print(auto_ARIMA_model.best_model.model_summary_series)
print(auto_ARIMA_model.best_model.arma_coefficients_summary_df)
# =====================================================================================================================

