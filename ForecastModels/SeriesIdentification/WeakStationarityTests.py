import os
import pandas as pd
import statsmodels.api as sm_
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt


from ForecastModels.ModelOne.ARIMAmodel import ARIMAmodel
from ForecastModels.DataProcessTools.dataProcessTools import load_FRED_data_df, compute_log_returns, \
    compute_simple_returns

from ForecastModels.DataProcessTools.dataSubSets import create_sub_samples


class WeakStationarityTests(object):
    def __init__(self, time_series, size=100):
        self.time_series = time_series.copy()
        self.size = size
        self.sub_series = create_sub_samples(time_series, size)

        self.time_series_stats = None
        self.compute_time_series_stats()
        self.sub_series_stats = None
        self.compute_sub_series_stats()

        self.is_stationary = self.sub_series_stats['is_mean_constant'].all() and self.sub_series_stats['is_acf_constant'].all()

    def compute_time_series_stats(self):
        self.time_series_stats = pd.Series(name="time_series_stats")
        self.time_series_stats["mean"] = self.time_series.mean()
        self.time_series_stats["std"] = self.time_series.std()

        acf = sm_.tsa.stattools.acf(self.time_series, nlags=1, qstat=False)
        std = ((1 + 2 * acf[1:].sum()) / len(self.time_series)) ** 0.5
        print(std)
        self.time_series_stats["lag_1_acf"] = acf[1]
        self.time_series_stats["lag_1_acf_std"] = std

    def compute_sub_series_stats(self):
        self.sub_series_stats = pd.DataFrame()
        for date, time_serie in self.sub_series.items():
            sub_serie_stat = pd.Series(name=date)
            sub_serie_stat["mean"] = time_serie.mean()
            sub_serie_stat["std"] = time_serie.std()

            mean_t_statistic, mean_p_value = self.eval_sub_series_stationarity(
                mean_1=self.time_series_stats["mean"],
                std_1=self.time_series_stats["std"],
                mean_2=sub_serie_stat["mean"],
                std_2=sub_serie_stat["std"],
                n_1=len(self.time_series_stats),
                n_2=len(time_serie))

            sub_serie_stat["mean_t_statistic"] = mean_t_statistic
            sub_serie_stat["mean_p_value"] = mean_p_value
            sub_serie_stat["is_mean_constant"] = mean_p_value > 5/100

            acf = sm_.tsa.stattools.acf(time_serie, nlags=1, qstat=False)
            std = ((1 + 2 * acf[1:].sum()) / len(time_serie)) ** 0.5
            sub_serie_stat["lag_1_acf"] = acf[1]
            sub_serie_stat["lag_1_acf_std"] = std

            acf_t_statistic, acf_p_value = self.eval_sub_series_stationarity(
                mean_1=self.time_series_stats["lag_1_acf"],
                std_1=self.time_series_stats["lag_1_acf_std"],
                mean_2=sub_serie_stat["lag_1_acf"],
                std_2=sub_serie_stat["lag_1_acf_std"],
                n_1=len(self.time_series_stats),
                n_2=len(time_serie))

            sub_serie_stat["acf_t_statistic"] = acf_t_statistic
            sub_serie_stat["acf_p_value"] = acf_p_value
            sub_serie_stat["is_acf_constant"] = acf_p_value > 5/100

            self.sub_series_stats = self.sub_series_stats.append(sub_serie_stat.copy())

    def eval_sub_series_stationarity(self, mean_1, mean_2, std_1, std_2, n_1, n_2):
        std_1_adj = std_1 / np.sqrt(n_1)
        std_2_adj = std_2 / np.sqrt(n_2)
        t_statistic = (mean_1 - mean_2) / np.sqrt(std_1_adj ** 2 + std_2_adj ** 2)
        df = (std_1_adj ** 2 + std_2_adj ** 2) ** 2 / (std_1_adj ** 4 / (n_1 - 1) + std_2_adj ** 4 / (n_2 - 1))
        p_value = 2 * (1 - st.t(df=df).cdf(np.abs(t_statistic)))
        return t_statistic, p_value #, df

    def plot_series_acf_t(self, level_val=1.0):
        fig, axs = plt.subplots(2, 1, figsize=(3, 9), sharex=True)
        axs[0].plot(self.sub_series_stats["lag_1_acf"],  'o-')
        axs[0].plot(self.sub_series_stats["lag_1_acf"] + level_val*self.sub_series_stats["lag_1_acf_std"], 'b--')
        axs[0].plot(self.sub_series_stats["lag_1_acf"] - level_val*self.sub_series_stats["lag_1_acf_std"], 'b--')
        axs[0].hlines(y=self.time_series_stats["lag_1_acf"],
                      xmin=self.sub_series_stats["lag_1_acf"].index[0],
                      xmax=self.sub_series_stats["lag_1_acf"].index[-1],
                      colors='r')
        axs[0].hlines(y=self.time_series_stats["lag_1_acf"] + level_val*self.time_series_stats["lag_1_acf_std"],
                      xmin=self.sub_series_stats["lag_1_acf"].index[0],
                      xmax=self.sub_series_stats["lag_1_acf"].index[-1],
                      colors='r',
                      linestyles='dashed')
        axs[0].hlines(y=self.time_series_stats["lag_1_acf"] - level_val*self.time_series_stats["lag_1_acf_std"],
                      xmin=self.sub_series_stats["lag_1_acf"].index[0],
                      xmax=self.sub_series_stats["lag_1_acf"].index[-1],
                      colors='r',
                      linestyles='dashed')
        axs[1].plot(self.sub_series_stats["acf_t_statistic"],  'o-')
        axs[1].hlines(y=1.96,
                      xmin=self.sub_series_stats["lag_1_acf"].index[0],
                      xmax=self.sub_series_stats["lag_1_acf"].index[-1],
                      colors='m',
                      linestyles='dashed')
        axs[1].hlines(y=-1.96,
                      xmin=self.sub_series_stats["lag_1_acf"].index[0],
                      xmax=self.sub_series_stats["lag_1_acf"].index[-1],
                      colors='m',
                      linestyles='dashed')

    def plot_series_mean_t(self, level_val=1.0):
        fig, axs = plt.subplots(2, 1, figsize=(3, 9), sharex=True)
        axs[0].plot(self.sub_series_stats["mean"],  'o-')
        axs[0].plot(self.sub_series_stats["mean"] + level_val*self.sub_series_stats["std"], 'b--')
        axs[0].plot(self.sub_series_stats["mean"] - level_val*self.sub_series_stats["std"], 'b--')
        axs[0].hlines(y=self.time_series_stats["mean"],
                      xmin=self.sub_series_stats["mean"].index[0],
                      xmax=self.sub_series_stats["mean"].index[-1],
                      colors='r')
        axs[0].hlines(y=self.time_series_stats["mean"] + level_val*self.time_series_stats["std"],
                      xmin=self.sub_series_stats["mean"].index[0],
                      xmax=self.sub_series_stats["mean"].index[-1],
                      colors='r',
                      linestyles='dashed')
        axs[0].hlines(y=self.time_series_stats["mean"] - level_val*self.time_series_stats["std"],
                      xmin=self.sub_series_stats["mean"].index[0],
                      xmax=self.sub_series_stats["mean"].index[-1],
                      colors='r',
                      linestyles='dashed')
        axs[1].plot(self.sub_series_stats["mean_t_statistic"],  'o-')
        axs[1].hlines(y=1.96,
                      xmin=self.sub_series_stats["mean"].index[0],
                      xmax=self.sub_series_stats["mean"].index[-1],
                      colors='m',
                      linestyles='dashed')
        axs[1].hlines(y=-1.96,
                      xmin=self.sub_series_stats["mean"].index[0],
                      xmax=self.sub_series_stats["mean"].index[-1],
                      colors='m',
                      linestyles='dashed')


if __name__ == '__main__':
    # Loadding of Data ===================================================================================================
    path_default = os.path.dirname(os.path.dirname(os.getcwd())) + '/data'
    # --------------------------------------------------------------
    path_module3_data = path_default + "/IMFx/Module3_data/"
    file_name = "module3_data_PE_Ratios.csv"

    module3_data_PE_Rations_df = pd.read_csv(path_module3_data + file_name)
    module3_data_PE_Rations_df = module3_data_PE_Rations_df.set_index(['dateid'])
    pe_saf_df = module3_data_PE_Rations_df['pe_saf'].dropna()

    # --------------------------------------------------------------
    path_default_FRED = path_default + '/FRED'
    path_GDP_GNP = path_default_FRED + "/GDP_GNP"
    date_threshold_max_ = pd.to_datetime("2020-01-01 20:00:00")
    GDP_df = load_FRED_data_df(file_name='GDP.csv', path=path_GDP_GNP, date_threshold_max=date_threshold_max_)
    GDP_series = GDP_df["GDP"]
    GDP_log_ret = compute_log_returns(GDP_series)
    # --------------------------------------------------------------
    # ====================================================================================================================

    wst = WeakStationarityTests(pe_saf_df)
    wst.plot_series_mean_t()
    wst.plot_series_acf_t()

# TODO:
#   Validate mean, var, autoccor remains statistically const:
#       Compare subsamples stats vs all data stats
