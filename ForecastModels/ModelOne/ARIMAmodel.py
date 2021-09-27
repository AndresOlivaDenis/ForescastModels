import os
import scipy.stats as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm_
from ForecastModels.ModelOne.toDos.DataTransformations import DataTransformations
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.arima.model import ARIMA
import warnings


class ARIMAmodel(object):
    def __init__(self, data_df, order, seasonal_order=(0, 0, 0, 0)):
        warnings.filterwarnings("ignore", category=ValueWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        self.data_df = data_df.copy()
        self.order = order
        self.seasonal_order = seasonal_order
        self.name = "{}".format(order)
        if np.any(seasonal_order):
            self.name += " {}".format(seasonal_order)

        self.correlogram_df = ARIMAmodel.get_correlogram_df(data_df=data_df)
        self.arma_mod, self.arma_res = ARIMAmodel.fit_ARIMA_model(data_df=data_df, order=order, seasonal_order=seasonal_order)

        self.residual_checks = ARIMAmodel.make_residual_checks(self.arma_res, series_name=self.name)
        self.residuals_correlogram_df = ARIMAmodel.get_correlogram_df(data_df=self.arma_res.resid)
        self.model_summary_series, self.arma_coefficients_summary_df = ARIMAmodel.make_arma_fit_summary(self.arma_res, series_name=self.name)
        self.model_ok = self.residual_checks['ll_white_noise_residuals'] and self.model_summary_series['has_significant_predictors_coefficients']

        self.model_summary_series['ll_white_noise_residuals'] = self.residual_checks['ll_white_noise_residuals']
        self.model_summary_series['model_ok'] = self.model_ok

    # Identification methods -----------------------------------------------------------------------------------------
    def plot_correlograms(self, lags=20, alpha=5. / 100):
        ARIMAmodel.plot_correlogram(self.data_df, lags=lags, alpha=alpha)

    def plot_residuals_correlograms(self, lags=20, alpha=5. / 100):
        ARIMAmodel.plot_correlogram(self.arma_res.resid, lags=lags, alpha=alpha)

    def plot_residuals_dignostic(self):
        self.arma_res.plot_diagnostics()

    def print_arma_res_summary(self):
        print(self.arma_res.summary())

    # ----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def plot_correlogram(data_df, lags=20, alpha=5. / 100):
        DataTransformations.get_correlogram_df(data_df, lags=lags, alpha=alpha)

    @staticmethod
    def get_correlogram_df(data_df, lags=20, alpha=5. / 100):
        return DataTransformations.get_correlogram_df(data_df, lags=lags, alpha=alpha)

    @staticmethod
    def fit_ARIMA_model(data_df, order, seasonal_order=(0, 0, 0, 0)):
        arma_mod = ARIMA(data_df, order=order, seasonal_order=seasonal_order, trend=None)
        arma_res = arma_mod.fit()
        return arma_mod, arma_res

    @staticmethod
    def make_residual_checks(arma_res, series_name=None):
        residual_checks_dict = dict()

        T = len(arma_res.resid)
        dist = st.t(df=T - 1)
        t_val = np.abs(arma_res.resid.mean()/arma_res.resid.std())
        p_val = 2 * (1 - dist.cdf(t_val))
        residual_checks_dict['zero_mean_t_val'] = t_val
        residual_checks_dict['zero_mean_p_val'] = p_val
        residual_checks_dict['zero_mean_null_hypothesis_reject'] = p_val < 5. / 100

        test_serial_correlation = arma_res.test_serial_correlation(method='ljungbox')
        statistic_res, pvalues_res = test_serial_correlation[0][0], test_serial_correlation[0][1]
        residual_checks_dict['min_p_value'] = pvalues_res.min()
        residual_checks_dict['max_p_value'] = pvalues_res.max()
        residual_checks_dict['null_hypothesis_reject'] = pvalues_res.min() < 5. / 100

        residual_checks_dict['ll_white_noise_residuals'] = not residual_checks_dict['null_hypothesis_reject'] and not residual_checks_dict['zero_mean_null_hypothesis_reject']
        return pd.Series(residual_checks_dict, name=series_name)

    @staticmethod
    def make_arma_fit_summary(arma_res, series_name=None):
        coefficients_summary_df = pd.DataFrame(index=arma_res.param_names)
        coefficients_summary_df['coefficient'] = arma_res.params.copy()
        coefficients_summary_df['tvalues'] = arma_res.tvalues.copy()
        coefficients_summary_df['pvalues'] = arma_res.pvalues.copy()
        # coefficients_summary_df['std Error'] # TODO_
        coefficients_summary_df['null_hypothesis_reject'] = arma_res.pvalues < 5. / 100

        arma_coefficients_summary_df = coefficients_summary_df.copy()
        arma_coefficients_summary_df = arma_coefficients_summary_df.drop(index=["const", "sigma2"])

        data_df = arma_res.fittedvalues + arma_res.resid
        T = len(data_df)
        model_summary_dict = dict()

        TSS = ((data_df - data_df.mean()) ** 2).sum()
        model_summary_dict['R_squared'] = 1 - arma_res.sse/TSS
        model_summary_dict['R_squared_adj'] = 1 - (1 - model_summary_dict['R_squared'])*(T - 1)/(T - (arma_res.df_model - 2) - 1)
        model_summary_dict['S.E of regression'] = arma_res.params.sigma2**0.5
        model_summary_dict['BIC'] = -2.0 / T * arma_res.llf + arma_res.df_model * np.log(T) / T  # Schwarz criterion
        model_summary_dict['AIC'] = -2.0 / T * arma_res.llf + 2 / T * arma_res.df_model  # AIC
        model_summary_dict['BIC'] = -2.0 / T * arma_res.llf + arma_res.df_model * np.log(T) / T  # Schwarz criterion
        # model_summary_dict['Durbin_Watson_statistic'] = (arma_res.resid.diff().dropna() ** 2).sum() / (
        #             arma_res.resid ** 2).sum()

        model_summary_dict['min_coeff_pvalue'] = arma_coefficients_summary_df['pvalues'].min()
        model_summary_dict['max_coeff_pvalue'] = arma_coefficients_summary_df['pvalues'].max()
        model_summary_dict['n_significant_predictors_coefficients'] = arma_coefficients_summary_df['null_hypothesis_reject'].sum()
        model_summary_dict['n_not_significant_predictors_coefficients'] = (~arma_coefficients_summary_df['null_hypothesis_reject']).sum()
        model_summary_dict['has_significant_predictors_coefficients'] = arma_coefficients_summary_df[
            'null_hypothesis_reject'].any()

        model_summary_series = pd.Series(model_summary_dict, name=series_name)
        return model_summary_series, coefficients_summary_df
    # ----------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    # Loadding of Data ===================================================================================================
    path_default = os.path.dirname(os.path.dirname(os.getcwd())) + '/data'
    path_module3_data = path_default + "/IMFx/Module3_data/"
    file_name = "module3_data_PE_Ratios.csv"

    module3_data_PE_Rations_df = pd.read_csv(path_module3_data + file_name)
    module3_data_PE_Rations_df = module3_data_PE_Rations_df.set_index(['dateid'])
    pe_saf_df = module3_data_PE_Rations_df['pe_saf'].dropna()
    # ====================================================================================================================

    correlogram_df_pe_saf_df = ARIMAmodel.get_correlogram_df(data_df=pe_saf_df)

    arima_model = ARIMAmodel(data_df=pe_saf_df, order=(4, 0, 0), seasonal_order=(0, 0, 0, 0))
    arima_model_6 = ARIMAmodel(data_df=pe_saf_df, order=(6, 0, 0), seasonal_order=(0, 0, 0, 0))
    print(arima_model_6.model_summary_series)
    print(arima_model_6.arma_coefficients_summary_df)

    arima_model_6_1 = ARIMAmodel(data_df=pe_saf_df, order=(6, 0, 1), seasonal_order=(0, 0, 0, 0))
