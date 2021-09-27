import pandas as pd
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss


class StationarityTests(object):
    def __init__(self, time_series, cl=5/100):
        self.stationarity_summary = None
        self.cl = cl
        self.adf_test_output = StationarityTests.adf_test(time_series)
        self.kpss_test_output = StationarityTests.kpss_test(time_series)

        self.run_summary()

    def run_summary(self):
        self.stationarity_summary = dict()
        self.stationarity_summary["adf_p_value"] = self.adf_test_output["p-value"]
        self.stationarity_summary["adf_null_hypothesis_reject"] = self.adf_test_output["p-value"] < self.cl
        self.stationarity_summary["adf_stationarity_evidence"] = self.stationarity_summary["adf_null_hypothesis_reject"]

        self.stationarity_summary["kpss_p_value"] = self.kpss_test_output["p-value"]
        self.stationarity_summary["kpss_null_hypothesis_reject"] = self.kpss_test_output["p-value"] < self.cl
        self.stationarity_summary["kpss_stationarity_evidence"] = not self.stationarity_summary["kpss_null_hypothesis_reject"]

        if self.stationarity_summary["adf_stationarity_evidence"]:
            if self.stationarity_summary["kpss_stationarity_evidence"]:
                self.stationarity_summary["is_stationary"] = True
                self.stationarity_summary["is_not_stationary"] = False
                self.stationarity_summary["conclusion"] = "Both tests conclude that the series is stationary " \
                                                          "- The series is stationary"
            else:
                self.stationarity_summary["is_stationary"] = False
                self.stationarity_summary["is_not_stationary"] = False
                self.stationarity_summary["conclusion"] = "The series is difference stationary. " \
                                                          "Differencing is to be used to make series stationary. " \
                                                          "The differenced series is checked for stationarity."
        else:
            if self.stationarity_summary["kpss_stationarity_evidence"]:
                self.stationarity_summary["is_stationary"] = False
                self.stationarity_summary["is_not_stationary"] = False
                self.stationarity_summary["conclusion"] = "The series is trend stationary. " \
                                                          "Trend needs to be removed to make series strict stationary. " \
                                                          "The detrended series is checked for stationarity"
            else:
                self.stationarity_summary["is_stationary"] = False
                self.stationarity_summary["is_not_stationary"] = True
                self.stationarity_summary["conclusion"] = "Both tests conclude that the series is not stationary " \
                                                          "- The series is not stationary"

        self.stationarity_summary = pd.Series(self.stationarity_summary)

    @staticmethod
    def adf_test(time_series):
        dftest = adfuller(time_series, autolag="AIC")
        dfoutput = pd.Series(dftest[0:4],
                             index=["Test Statistic", "p-value", "#Lags Used", "Number of Observations Used"])
        return dfoutput

    @staticmethod
    def kpss_test(time_series):
        kpsstest = kpss(time_series, regression="c", nlags="auto")
        kpss_output = pd.Series(kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"])
        return kpss_output

# https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity_detrending_adf_kpss.html


if __name__ == '__main__':
    # Loadding of Data ===================================================================================================
    path_default = os.path.dirname(os.path.dirname(os.getcwd())) + '/data'
    path_module3_data = path_default + "/IMFx/Module3_data/"
    file_name = "module3_data_PE_Ratios.csv"

    module3_data_PE_Rations_df = pd.read_csv(path_module3_data + file_name)
    module3_data_PE_Rations_df = module3_data_PE_Rations_df.set_index(['dateid'])
    pe_saf_df = module3_data_PE_Rations_df['pe_saf'].dropna()
    # ====================================================================================================================

    stationarity_test = StationarityTests(pe_saf_df)
    print(stationarity_test.stationarity_summary)

    # Loadding of Data ===================================================================================================
    file_name_reer = "module3_data_REER.csv"

    module3_data_REER_df = pd.read_csv(path_module3_data + file_name_reer)
    module3_data_REER_df = module3_data_REER_df.set_index(['dateid'])
    reer_mys_df = module3_data_REER_df['reer_mys'].dropna()
    # ====================================================================================================================

    stationarity_test_reer= StationarityTests(reer_mys_df)
    print(stationarity_test_reer.stationarity_summary)