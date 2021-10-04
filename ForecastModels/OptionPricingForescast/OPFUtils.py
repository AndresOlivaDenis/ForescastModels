import os
import pandas as pd

from SimulationScenarioEngine.DataPreprocesor.dataPreprocess import DataPreprocess
from SimulationScenarioEngine.ReturnsGenerators.returnsGeneratorDistributionFit import ReturnsGeneratorDistributionFit
from SimulationScenarioEngine.ReturnsGenerators.returnsGeneratorHistDataSample import ReturnsGeneratorHistDataSample

default_distribution_fitter_params = dict(cost_function="MLE",
                                          distribution_list=["Normal",
                                                             # "T"    #
                                                             ],
                                          use_fit_method=True)


def build_return_generator_by_distribution_fit(data_df, distribution_fitter_params=default_distribution_fitter_params):
    rgdf = ReturnsGeneratorDistributionFit(data_df=data_df, returns_generator_params_dict=dict())
    rgdf.fit_distribution_to_data(distribution_fitter_params=distribution_fitter_params)

    rgdf.compute_moments_from_sample(sample_size=50000)
    rgdf.compute_daily_moments_from_sample(daily_sample_size=50000)
    rgdf.compute_yearly_moments_from_sample(yearly_sample_size=10000)

    return rgdf


def build_return_generator_by_hist_data_sample(data_df):
    rghd = ReturnsGeneratorHistDataSample(data_df=data_df, returns_generator_params_dict=dict())

    rghd.compute_moments_from_sample(sample_size=10000)
    rghd.compute_daily_moments_from_sample(daily_sample_size=10000)
    rghd.compute_yearly_moments_from_sample(yearly_sample_size=10000)

    return rghd