import pandas as pd
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm_

# Loadding of Data ===================================================================================================
path_default = os.path.dirname(os.path.dirname(os.getcwd())) + '/data'
path_module3_data = path_default + "/IMFx/Module3_data/"
file_name = "module3_data_PE_Ratios.csv"

module3_data_PE_Rations_df = pd.read_csv(path_module3_data + file_name)
module3_data_PE_Rations_df = module3_data_PE_Rations_df.set_index(['dateid'])
pe_saf_df = module3_data_PE_Rations_df['pe_saf'].dropna()
# ====================================================================================================================

# Plotting of data ====================================================================================================
pe_saf_df.plot()
plt.grid()
# =====================================================================================================================

lags = 20

# Plotting and retrieving ACF =========================================================================================
fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True)

f = sm_.graphics.tsa.plot_acf(pe_saf_df, lags=lags, ax=ax1)
acf, acf_confint, acf_qstat, acf_pvalues = sm_.tsa.stattools.acf(pe_saf_df, nlags=lags, qstat=True, alpha=5. / 100)
# =====================================================================================================================

# Plotting and retrieving PACF ========================================================================================
f = sm_.graphics.tsa.plot_pacf(pe_saf_df, lags=lags, ax=ax2)
pacf, pacf_confint = sm_.tsa.stattools.pacf(pe_saf_df, nlags=lags, alpha=5. / 100)
# =====================================================================================================================

# ACF & PACF dataFrame ================================================================================================
acf_pacf_dict = dict(lag=list(range(1, lags + 1)),
                     acf=acf[1:],
                     pacf=pacf[1:],
                     qstat=acf_qstat,
                     pvalues=acf_pvalues)
correlogram_df = pd.DataFrame(acf_pacf_dict)
alpha=5. / 100
correlogram_df["null_hypothesis_reject"] = correlogram_df["pvalues"] > alpha
# ====================================================================================================================
