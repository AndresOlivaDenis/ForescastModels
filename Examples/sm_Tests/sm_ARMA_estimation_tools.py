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
# ====================================================================================================================

# ARMA estimate =========================================================================================

# Fit and estimate --------------------------------------

# https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html#statsmodels.tsa.arima.model.ARIMA
arma_mod = ARIMA(pe_saf_df, order=(4, 0, 0), seasonal_order=(0, 0, 0, 0), trend=None)
arma_res = arma_mod.fit()

print(arma_res.summary())

# -------------------------------------------------------

# Residuals checks --------------------------------------
# residuals = arma_res.resid.copy()
# lags = arma_res.df_model + 5
# f = sm_.graphics.tsa.plot_acf(residuals, lags=lags)
# acf_residuals, acf_confint_residuals, qstat_residuals, pvalues_residuals = sm_.tsa.stattools.acf(residuals, nlags=lags, qstat=True, alpha=5. / 100)
res_mean = arma_res.resid.mean()
res_mean_var_rel = arma_res.resid.mean()/arma_res.resid.std()

test_serial_correlation = arma_res.test_serial_correlation(method='ljungbox')   # https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.mlemodel.MLEResults.test_serial_correlation.html#statsmodels.tsa.statespace.mlemodel.MLEResults.test_serial_correlation
statistic_res, pvalues_res = test_serial_correlation[0][0], test_serial_correlation[0][1]

test_normality = arma_res.test_normality(method='jarquebera')
JB, JBpv, skew, kurtosis = test_normality[0][0], test_normality[0][1], test_normality[0][2], test_normality[0][3]
# arma_res.plot_diagnostics()
# -------------------------------------------------------

# Other fitting computations ----------------------------
reg_SE = arma_res.params.sigma2**0.5     # standard error of the regression
SSR = arma_res.sse

T = len(pe_saf_df)
AIC = -2.0/T*arma_res.llf + 2/T*arma_res.df_model # AIC
BIC = -2.0/T*arma_res.llf + arma_res.df_model*np.log(T)/T   # Schwarz criterion

TSS = ((pe_saf_df - pe_saf_df.mean())**2).sum()
R_squared = 1 - arma_res.sse/TSS    # R-squared
df_model = arma_res.df_model
R_squared_adj = 1 - (1 - R_squared)*(T - 1)/(T - (df_model - 2) - 1)    # Adjusted R-squared

resi = pe_saf_df.values - arma_res.forecasts
# -------------------------------------------------------
Durbin_Watson_statistic = (arma_res.resid.diff().dropna()**2).sum()/(arma_res.resid**2).sum()
# =======================================================================================================
