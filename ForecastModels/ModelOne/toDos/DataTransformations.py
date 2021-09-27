import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm_


class DataTransformations(object):
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

# TODO:
#   Apply data Transformations:
#       Stabilize the variance: Log transformations
#       differencing transformations: "standard" and "seasonals"
#       Method to "transform back" data
#           i.e: use a reference value (by default last one) and apply inverse of applied transformations
#       Make autotransformations ? :
#           Transform until it appears to be stationary
#           https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html (see notes)
#       Make data plotting definitions (original & transformed (and maybe even by steps)
#       Include here methods: plot_correlogram, get_correlogram_df from ARIMAmodel.py
#       Detecting seasonality
#   Could be usefull as reference: https://otexts.com/fpp2/arima-r.html#arima-r

#   Refactor to new package (its usefull also for model 2
#   When traying transformation follow a "consistent" order:
#       Try first with log transformations and diff 1 and diff 2 ect ...
#       Stop on first transformation that is stationary!
