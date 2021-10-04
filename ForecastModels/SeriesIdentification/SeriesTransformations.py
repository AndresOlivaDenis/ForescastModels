import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm_


class SeriesTransformations(object):
    def __init__(self, time_series):
        self.time_series = time_series
        self.transformed_time_series = None
        self.applied_transformation = None  # Define an ENUM ?
        pass

    @staticmethod
    def plot_correlogram(data_df, lags=20, alpha=5. / 100):
        fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True)
        sm_.graphics.tsa.plot_acf(data_df, lags=lags, ax=ax1, alpha=alpha)
        sm_.graphics.tsa.plot_pacf(data_df, lags=lags, ax=ax2, alpha=alpha)

    @staticmethod
    def get_correlogram_df(data_df, lags=20, alpha=5. / 100):
        acf, acf_confint, acf_qstat, acf_pvalues = sm_.tsa.stattools.acf(data_df, nlags=lags, qstat=True, alpha=alpha)
        pacf, pacf_confint = sm_.tsa.stattools.pacf(data_df, nlags=lags, alpha=alpha)
        acf_pacf_dict = dict(lag=list(range(1, lags + 1)),
                             acf=acf[1:],
                             pacf=pacf[1:],
                             qstat=acf_qstat,
                             pvalues=acf_pvalues)
        correlogram_df = pd.DataFrame(acf_pacf_dict)
        correlogram_df["null_hypothesis_reject"] = correlogram_df["pvalues"] < alpha
        return correlogram_df

# CURRENTLY!:
# END this & create a notebook example of making forecast (to later recap, when see chapter 4)
# TODO 1:
#   Apply data Transformations:
#       Make it more normal & Stabilize the variance: Log and & Box-Cox transformations
#           For now maybe just stuck with log transformation
#       Make data stationary: differencing transformations: "standard" and "seasonals"
#       Method to "transform back" data
#           i.e: use a reference value (by default last one) and apply inverse of applied transformations
#
# TODO 2:
#   Make data plotting definitions (original & transformed (and maybe even by steps)
#
# TODO 3:
#   Make autotransformations ? :
#           Transform until it appears to be stationary
#           https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html (see notes)
#   Detecting seasonality
#   Could be usefull as reference: https://otexts.com/fpp2/arima-r.html#arima-r

#   Refactor to new package (its usefull also for model 2
#   When traying Automate transformation follow a "consistent" order:
#       Try first with log transformations and diff 1 and diff 2 ect ...
#       Stop on first transformation that is stationary!
