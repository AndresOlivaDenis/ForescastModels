import pandas as pd
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm_
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


# Loadding of Data ===================================================================================================
path_default = os.path.dirname(os.path.dirname(os.getcwd())) + '/data'
path_module3_data = path_default + "/IMFx/Module3_data/"
file_name = "module3_data_PE_Ratios.csv"

module3_data_PE_Rations_df = pd.read_csv(path_module3_data + file_name)
module3_data_PE_Rations_df = module3_data_PE_Rations_df.set_index(['dateid'])
pe_saf_df = module3_data_PE_Rations_df['pe_saf'].dropna()
# pe_saf_df = np.log(pe_saf_df)
pe_saf_df_diff = pe_saf_df.diff().dropna()
pe_saf_df_diff_diff = pe_saf_df.diff().diff().dropna()
# ====================================================================================================================

# ARMA estimate =========================================================================================

# Fit and estimate --------------------------------------

# https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html#statsmodels.tsa.arima.model.ARIMA
arma_mod = ARIMA(pe_saf_df, order=(4, 2, 0), seasonal_order=(0, 0, 0, 0), trend=None)
arma_res = arma_mod.fit()
arma_pred = arma_res.get_prediction()
print(arma_res.summary())

# https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html#statsmodels.tsa.arima.model.ARIMA
arma_mod_diff = ARIMA(pe_saf_df_diff_diff, order=(4, 0, 0), seasonal_order=(0, 0, 0, 0), trend=None)
arma_mod_diff_res = arma_mod_diff.fit()

print(arma_mod_diff_res.summary())


print(arma_res.summary())

print(arma_res.forecast())
print(arma_mod_diff_res.forecast() + 2*pe_saf_df[-1] - pe_saf_df[-2])


