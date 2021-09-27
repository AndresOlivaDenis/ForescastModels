import os
import pandas as pd
import numpy as np


from ForecastModels.DataProcessTools.dataProcessTools import load_FRED_data_df, compute_log_returns, compute_simple_returns

path_default = os.path.dirname(os.path.dirname(os.getcwd())) + '/data'
path_default_FRED = path_default + '/FRED'
path_GDP_GNP = path_default_FRED + "/GDP_GNP"

date_threshold_max_ = pd.to_datetime("2020-01-01 20:00:00")

GDP_df = load_FRED_data_df(file_name='GDP.csv', path=path_GDP_GNP, date_threshold_max=date_threshold_max_)
GDP_log_ret = compute_log_returns(GDP_df)
GDP_simple_ret = compute_simple_returns(GDP_df)
GDP_simp_ret_q = np.exp(np.log(GDP_df).diff(periods=4).dropna()) - 1