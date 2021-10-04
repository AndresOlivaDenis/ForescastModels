import unittest
import os
import numpy as np
import pandas as pd
import random
import scipy.stats as st

from SimulationScenarioEngine.DataPreprocesor.dataPreprocess import DataPreprocess


class TestReturnGenerators(unittest.TestCase):

    def test_dates_stats(self):
        test_paths_and_expected_res = [dict(folder_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                     "Data/Data_IMC_P01_D1"),
                                            expected_frames_in_day=1),
                                       dict(folder_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                     "Data/Data_IMC_P01_H4"),
                                            expected_frames_in_day=24/4),
                                       dict(folder_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                     "Data/Data_IMC_P01_H2"),
                                            expected_frames_in_day=24/2),
                                       dict(folder_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                     "Data/Data_IMC_P01_M10"),
                                            expected_frames_in_day=1440/10 - 6*1.5)
                                       ]
        for test_case in test_paths_and_expected_res:
            folder_path = test_case["folder_path"]
            file_name = "USTEC.csv"

            load_file_args = dict(ref_colum_date="Time",
                                  ref_colum_price="adj Close",
                                  folder_path=folder_path)
            data_df = DataPreprocess.load_file_content(file_name=file_name, load_file_args=load_file_args)

            returns_generator_params_dict = dict(drop_out_of_date_median_values=True)
            rg = ReturnsGenerator(data_df=data_df,
                                  returns_generator_params_dict=returns_generator_params_dict)

            print("timedelta: {}, \n frames_in_a_day: {}, expected: {}".format(rg.timedelta,
                                                                               rg.frames_in_a_day,
                                                                               test_case["expected_frames_in_day"]))

            self.assertEqual(rg.frames_in_a_day, test_case["expected_frames_in_day"])

    def test_dates_stats_(self):
        test_paths_and_expected_res = [dict(folder_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                     "Data/Data_IMC_P01_D1"),
                                            expected_frames_in_day=1),
                                       dict(folder_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                     "Data/Data_IMC_P01_H4"),
                                            expected_frames_in_day=24/4),
                                       dict(folder_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                     "Data/Data_IMC_P01_H2"),
                                            expected_frames_in_day=24/2),
                                       dict(folder_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                     "Data/Data_IMC_P01_M10"),
                                            expected_frames_in_day=1440/10 - 6*1.5 + 2)
                                       ]
        for test_case in test_paths_and_expected_res:
            folder_path = test_case["folder_path"]
            file_name = "USTEC.csv"

            load_file_args = dict(ref_colum_date="Time",
                                  ref_colum_price="adj Close",
                                  folder_path=folder_path)
            data_df = DataPreprocess.load_file_content(file_name=file_name, load_file_args=load_file_args)

            returns_generator_params_dict = dict(drop_out_of_date_median_values=False)
            rg = ReturnsGenerator(data_df=data_df,
                                  returns_generator_params_dict=returns_generator_params_dict)

            print("timedelta: {}, \n frames_in_a_day: {}, expected: {}".format(rg.timedelta,
                                                                               rg.frames_in_a_day,
                                                                               test_case["expected_frames_in_day"]))

            self.assertEqual(rg.frames_in_a_day, test_case["expected_frames_in_day"])

    def test_ReturnsGenerator_compute_compute_data_moments_01(self):
        np.random.seed(0)
        loc, scale = 4, 10

        # Data creation  ---------------------------------------------
        folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data/Data_IMC_P01_M10")
        file_name = "USTEC.csv"

        load_file_args = dict(ref_colum_date="Time",
                              ref_colum_price="adj Close",
                              folder_path=folder_path)
        data_df = DataPreprocess.load_file_content(file_name=file_name, load_file_args=load_file_args)
        norm_obj = st.norm(loc=loc, scale=scale)
        data_df["USTEC"] = norm_obj.rvs(size=data_df.shape[0])
        # ----------------------------------------------------------------

        returns_generator_params_dict = dict(drop_out_of_date_median_values=False)
        rg = ReturnsGenerator(data_df=data_df,
                              returns_generator_params_dict=returns_generator_params_dict)

        mean, var, skew, kurt = norm_obj.stats(moments='mvsk')
        self.assertAlmostEqual(loc, rg.data_moments_stats['E'], delta=0.1)
        self.assertAlmostEqual(scale**2, rg.data_moments_stats['var'], delta=2)
        self.assertAlmostEqual(skew, rg.data_moments_stats['skew'], delta=0.1)
        self.assertAlmostEqual(kurt, rg.data_moments_stats['kurtosis'], delta=0.1)

        finaday = rg.frames_in_a_day
        self.assertAlmostEqual(loc*finaday, rg.data_moments_stats_daily_scaled['E'], delta=0.1*finaday)
        self.assertAlmostEqual(scale*finaday**0.5, rg.data_moments_stats_daily_scaled['std'], delta=2*finaday)

        finayear = rg.frames_in_a_day * rg.open_days_per_year
        self.assertAlmostEqual(loc*finayear, rg.data_moments_stats_yearly_scaled['E'], delta=0.1*finayear)
        self.assertAlmostEqual(scale*finayear**0.5, rg.data_moments_stats_yearly_scaled['std'], delta=2*finayear)

    def test_ReturnsGeneratorDistributionFit_compute_moments_from_sample_01(self):
        np.random.seed(0)
        loc, scale = 4, 10

        # Data creation  ---------------------------------------------
        # For dates references :
        folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data/Data_IMC_P01_M10")
        file_name = "USTEC.csv"

        load_file_args = dict(ref_colum_date="Time",
                              ref_colum_price="adj Close",
                              folder_path=folder_path)
        data_df = DataPreprocess.load_file_content(file_name=file_name, load_file_args=load_file_args)
        norm_obj = st.norm(loc=loc, scale=scale)
        data_df["USTEC"] = norm_obj.rvs(size=data_df.shape[0])
        # ----------------------------------------------------------------

        distribution_fitter_params = dict(cost_function="MethodOfMoments",  # "MLE",
                                          distribution_list=["Normal"],
                                          use_fit_method=True)
        returns_generator_params_dict = dict(drop_out_of_date_median_values=False)
        rgdf = ReturnsGeneratorDistributionFit(data_df=data_df,
                                               returns_generator_params_dict=returns_generator_params_dict)
        rgdf.fit_distribution_to_data(distribution_fitter_params=distribution_fitter_params)
        rgdf.compute_moments_from_sample()

        mean, var, skew, kurt = norm_obj.stats(moments='mvsk')
        self.assertAlmostEqual(mean, rgdf.sample_moments_stats['E'], delta=0.2)
        self.assertAlmostEqual(var, rgdf.sample_moments_stats['var'], delta=2)
        self.assertAlmostEqual(skew, rgdf.sample_moments_stats['skew'], delta=0.1)
        self.assertAlmostEqual(kurt, rgdf.sample_moments_stats['kurtosis'], delta=0.1)

        self.assertAlmostEqual(loc - 1.96*scale, rgdf.sample_moments_stats['ci_l'], delta=1)
        self.assertAlmostEqual(loc + 1.96*scale, rgdf.sample_moments_stats['ci_r'], delta=1)

        self.assertAlmostEqual(loc, rgdf.distribution_fitter.fitting_results['parameters']['Normal']['loc'],
                               delta=0.1)
        self.assertAlmostEqual(scale, rgdf.distribution_fitter.fitting_results['parameters']['Normal']['scale'],
                               delta=0.1)

        # ------------------------------------------------------------------------------------------
        days = 1000
        samples_paths = rgdf.generate_daily_returns_samples_paths(days)
        self.assertTupleEqual(samples_paths.shape, (days, rgdf.frames_in_a_day))
        self.assertAlmostEqual(mean, np.mean(samples_paths), delta=0.2)
        self.assertAlmostEqual(var, np.var(samples_paths), delta=2)

        rgdf.accept_joint_generation = False
        samples_paths_ = rgdf.generate_daily_returns_samples_paths(days)
        self.assertTupleEqual(samples_paths_.shape, (days, rgdf.frames_in_a_day))
        self.assertAlmostEqual(mean, np.mean(samples_paths_), delta=0.2)
        self.assertAlmostEqual(var, np.var(samples_paths_), delta=2)

        # ------------------------------------------------------------------------------------------
        rgdf.compute_daily_moments_from_sample(daily_sample_size=10000)

        std, finaday = var**0.5, rgdf.frames_in_a_day
        self.assertAlmostEqual(mean*finaday, rgdf.sample_daily_moments_stats['E'], delta=0.2*finaday)
        self.assertAlmostEqual(var*finaday, rgdf.sample_daily_moments_stats['var'], delta=5*finaday)
        self.assertAlmostEqual(std * finaday**0.5, rgdf.sample_daily_moments_stats['std'], delta=0.2*finaday)

        self.assertAlmostEqual(loc*finaday, rgdf.sample_daily_moments_stats['E'], delta=0.1*finaday)
        self.assertAlmostEqual(scale*finaday**0.5, rgdf.sample_daily_moments_stats['std'], delta=0.1*finaday)

        self.assertAlmostEqual(loc*finaday - 1.96*scale*finaday**0.5, rgdf.sample_daily_moments_stats['ci_l'],
                               delta=0.1*finaday)
        self.assertAlmostEqual(loc*finaday + 1.96*scale*finaday**0.5, rgdf.sample_daily_moments_stats['ci_r'],
                               delta=0.1*finaday)

        # ------------------------------------------------------------------------------------------
        rgdf.compute_yearly_moments_from_sample(yearly_sample_size=5000)

        std, finayear = var**0.5, rgdf.frames_in_a_day*rgdf.open_days_per_year
        self.assertAlmostEqual(mean*finayear, rgdf.sample_yearly_moments_stats['E'], delta=0.2*finayear)
        self.assertAlmostEqual(var*finayear, rgdf.sample_yearly_moments_stats['var'], delta=5*finayear)
        self.assertAlmostEqual(std * finayear**0.5, rgdf.sample_yearly_moments_stats['std'], delta=0.2*finayear)

        self.assertAlmostEqual(loc*finayear, rgdf.sample_yearly_moments_stats['E'], delta=0.1*finayear)
        self.assertAlmostEqual(scale*finayear**0.5, rgdf.sample_yearly_moments_stats['std'], delta=0.1*finayear)

        self.assertAlmostEqual(loc*finayear - 1.96*scale*finayear**0.5, rgdf.sample_yearly_moments_stats['ci_l'],
                               delta=0.2*finayear)
        self.assertAlmostEqual(loc*finayear + 1.96*scale*finayear**0.5, rgdf.sample_yearly_moments_stats['ci_r'],
                               delta=0.2*finayear)

        print(rgdf)

    def test_ReturnsGeneratorDistributionFit_adjust_location_for_distribution_obj_fitted_01(self):
        np.random.seed(0)
        loc, scale = 4, 10

        # Data creation  ---------------------------------------------
        # For dates references :
        folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data/Data_IMC_P01_H4")
        file_name = "USTEC.csv"

        load_file_args = dict(ref_colum_date="Time",
                              ref_colum_price="adj Close",
                              folder_path=folder_path)
        data_df = DataPreprocess.load_file_content(file_name=file_name, load_file_args=load_file_args)
        norm_obj = st.norm(loc=loc, scale=scale)
        data_df["USTEC"] = norm_obj.rvs(size=data_df.shape[0])
        # ----------------------------------------------------------------

        distribution_fitter_params = dict(cost_function="MLE",  # "MLE",
                                          distribution_list=["Normal", "T"], # ["Normal", "T"],
                                          use_fit_method=True)
        returns_generator_params_dict = dict(drop_out_of_date_median_values=False)
        rgdf = ReturnsGeneratorDistributionFit(data_df=data_df,
                                               returns_generator_params_dict=returns_generator_params_dict)
        rgdf.fit_distribution_to_data(distribution_fitter_params=distribution_fitter_params)

        sms, returns_samples = rgdf.compute_moments_from_sample(sample_size=50000)
        sds, cum_returns = rgdf.compute_daily_moments_from_sample(daily_sample_size=50000)
        sys, year_cum_returns = rgdf.compute_yearly_moments_from_sample(yearly_sample_size=10000)

        year_loc_val = 3000
        rgdf.adjust_location_for_distribution_obj_fitted(loc_val=year_loc_val)
        loc_adjustment = year_loc_val / (rgdf.open_days_per_year * rgdf.frames_in_a_day)
        for i in range(len(rgdf.distribution_fitter.fitting_results)):
            self.assertEqual(rgdf.distribution_fitter.fitting_results.iloc[i].parameters['loc'],
                             loc_adjustment)
            self.assertEqual(rgdf.distribution_fitter.fitting_results.iloc[i].distribution_obj_fitted.kwds['loc'],
                             loc_adjustment)
        sms_, returns_samples = rgdf.compute_moments_from_sample(sample_size=50000)
        sds_, cum_returns = rgdf.compute_daily_moments_from_sample(daily_sample_size=50000)
        sys_, year_cum_returns = rgdf.compute_yearly_moments_from_sample(yearly_sample_size=10000)

        self.assertAlmostEqual(sms_['E'], loc_adjustment, delta=0.2)
        self.assertAlmostEqual(sms_['std'], sms['std'], delta=0.2)

        self.assertAlmostEqual(sds_['E'], year_loc_val/rgdf.open_days_per_year, delta=0.5)
        self.assertAlmostEqual(sds_['std'], sds['std'], delta=0.5)

        self.assertAlmostEqual(sys_['E'], year_loc_val, delta=10)
        self.assertAlmostEqual(sys_['std'], sys['std'], delta=10)

        print("sms_['E']: {}, expected: {}".format(sms_['E'], loc_adjustment))
        print("sds_['E']: {}, expected: {}".format(sds_['E'], year_loc_val/rgdf.open_days_per_year))
        print("sys_['E']: {}, expected: {}".format(sys_['E'], year_loc_val))

        print("sms_['std']: {}, sms['std']: {}".format(sms_['std'], sms['std']))
        print("sds_['std']: {}, sds['std']: {}".format(sds_['std'], sds['std']))
        print("sys_['std']: {}, sys['std']: {}".format(sys_['std'], sys['std']))

    def test_ReturnsGeneratorDistributionFit_adjust_location_for_distribution_obj_fitted_02(self):
        np.random.seed(0)
        loc, scale = 4, 10

        # Data creation  ---------------------------------------------
        # For dates references :
        folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data/Data_IMC_P01_H4")
        file_name = "USTEC.csv"

        load_file_args = dict(ref_colum_date="Time",
                              ref_colum_price="adj Close",
                              folder_path=folder_path)
        data_df = DataPreprocess.load_file_content(file_name=file_name, load_file_args=load_file_args)
        t_obj = st.t(loc=loc, scale=scale, df=4.5)
        data_df["USTEC"] = t_obj.rvs(size=data_df.shape[0])
        # ----------------------------------------------------------------

        distribution_fitter_params = dict(cost_function="MLE",  # "MLE",
                                          distribution_list=["Normal", "T"], # ["Normal", "T"],
                                          use_fit_method=True)
        returns_generator_params_dict = dict(drop_out_of_date_median_values=False)
        rgdf = ReturnsGeneratorDistributionFit(data_df=data_df,
                                               returns_generator_params_dict=returns_generator_params_dict)
        rgdf.fit_distribution_to_data(distribution_fitter_params=distribution_fitter_params)

        sms, returns_samples = rgdf.compute_moments_from_sample(sample_size=50000)
        sds, cum_returns = rgdf.compute_daily_moments_from_sample(daily_sample_size=50000)
        sys, year_cum_returns = rgdf.compute_yearly_moments_from_sample(yearly_sample_size=10000)

        year_loc_val = 3000
        rgdf.adjust_location_for_distribution_obj_fitted(loc_val=year_loc_val)
        loc_adjustment = year_loc_val / (rgdf.open_days_per_year * rgdf.frames_in_a_day)
        for i in range(len(rgdf.distribution_fitter.fitting_results)):
            self.assertEqual(rgdf.distribution_fitter.fitting_results.iloc[i].parameters['loc'],
                             loc_adjustment)
            self.assertEqual(rgdf.distribution_fitter.fitting_results.iloc[i].distribution_obj_fitted.kwds['loc'],
                             loc_adjustment)
        sms_, returns_samples = rgdf.compute_moments_from_sample(sample_size=50000)
        sds_, cum_returns = rgdf.compute_daily_moments_from_sample(daily_sample_size=50000)
        sys_, year_cum_returns = rgdf.compute_yearly_moments_from_sample(yearly_sample_size=10000)

        self.assertAlmostEqual(sms_['E'], loc_adjustment, delta=0.2) # To much discrepancy
        self.assertAlmostEqual(sms_['std'], sms['std'], delta=0.2)

        self.assertAlmostEqual(sds_['E'], year_loc_val/rgdf.open_days_per_year, delta=0.5)
        self.assertAlmostEqual(sds_['std'], sds['std'], delta=0.5)

        self.assertAlmostEqual(sys_['E'], year_loc_val, delta=10)
        self.assertAlmostEqual(sys_['std'], sys['std'], delta=10)

        print("sms_['E']: {}, expected: {}".format(sms_['E'], loc_adjustment))
        print("sds_['E']: {}, expected: {}".format(sds_['E'], year_loc_val/rgdf.open_days_per_year))
        print("sys_['E']: {}, expected: {}".format(sys_['E'], year_loc_val))

        print("sms_['std']: {}, sms['std']: {}".format(sms_['std'], sms['std']))
        print("sds_['std']: {}, sds['std']: {}".format(sds_['std'], sds['std']))
        print("sys_['std']: {}, sys['std']: {}".format(sys_['std'], sys['std']))

    def test_ReturnsGeneratorDistributionFit_copy_obj_with_compute_change_scale_01(self):
        np.random.seed(0)
        loc, scale = 4, 10

        # Data creation  ---------------------------------------------
        # For dates references :
        folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data/Data_IMC_P01_D1")
        file_name = "USTEC.csv"

        load_file_args = dict(ref_colum_date="Time",
                              ref_colum_price="adj Close",
                              folder_path=folder_path)
        data_df = DataPreprocess.load_file_content(file_name=file_name, load_file_args=load_file_args)
        norm_obj = st.norm(loc=loc, scale=scale)
        data_df["USTEC"] = norm_obj.rvs(size=data_df.shape[0])
        # ----------------------------------------------------------------

        distribution_fitter_params = dict(cost_function="MLE",  # "MLE",
                                          distribution_list=["Normal", "T"],  # ["Normal", "T"],
                                          use_fit_method=True)
        returns_generator_params_dict = dict(drop_out_of_date_median_values=False)
        rgdf = ReturnsGeneratorDistributionFit(data_df=data_df,
                                               returns_generator_params_dict=returns_generator_params_dict)
        rgdf.fit_distribution_to_data(distribution_fitter_params=distribution_fitter_params)

        sms, returns_samples = rgdf.compute_moments_from_sample(sample_size=50000)
        sds, cum_returns = rgdf.compute_daily_moments_from_sample(daily_sample_size=50000)
        sys, year_cum_returns = rgdf.compute_yearly_moments_from_sample(yearly_sample_size=10000)

        timedelta_value = 1
        rgdf_2 = rgdf.copy_obj_with_compute_change_scale(timedelta_unit="h", timedelta_value=timedelta_value)

        self.assertEqual(rgdf_2.frames_in_a_day, 24)
        self.assertEqual(rgdf_2.generate_daily_returns_samples_paths(2).shape, (2, 24))

        for i in range(len(rgdf.distribution_fitter.fitting_results)):
            loc_scaling = rgdf.distribution_fitter.fitting_results.iloc[i].parameters['loc']
            loc_scaling *= (rgdf.frames_in_a_day/rgdf_2.frames_in_a_day)

            scale_scaling = rgdf.distribution_fitter.fitting_results.iloc[i].parameters['scale']
            scale_scaling *= (rgdf.frames_in_a_day / rgdf_2.frames_in_a_day)**0.5

            self.assertEqual(rgdf_2.distribution_fitter.fitting_results.iloc[i].parameters['loc'],
                             loc_scaling)
            self.assertEqual(rgdf_2.distribution_fitter.fitting_results.iloc[i].distribution_obj_fitted.kwds['loc'],
                             loc_scaling)
            self.assertEqual(rgdf_2.distribution_fitter.fitting_results.iloc[i].parameters['scale'],
                             scale_scaling)
            self.assertEqual(rgdf_2.distribution_fitter.fitting_results.iloc[i].distribution_obj_fitted.kwds['scale'],
                             scale_scaling)

        sms_, returns_samples = rgdf_2.compute_moments_from_sample(sample_size=50000)
        sds_, cum_returns = rgdf_2.compute_daily_moments_from_sample(daily_sample_size=50000)
        sys_, year_cum_returns = rgdf_2.compute_yearly_moments_from_sample(yearly_sample_size=10000)

        self.assertAlmostEqual(sds_['E'], sds['E'], delta=0.5)
        self.assertAlmostEqual(sds_['std'], sds['std'], delta=0.5)

        self.assertAlmostEqual(sys_['E'], sys['E'], delta=10)
        self.assertAlmostEqual(sys_['std'], sys['std'], delta=10)

        print("sms_['E']: {}, sms['E']: {}".format(sms_['E'], sms['E']))
        print("sds_['E']: {}, sds['E']: {}".format(sds_['E'], sds['E']))
        print("sys_['E']: {}, sys['E']: {}".format(sys_['E'], sys['E']))

        print("sms_['std']: {}, sms['std']: {}".format(sms_['std'], sms['std']))
        print("sds_['std']: {}, sds['std']: {}".format(sds_['std'], sds['std']))
        print("sys_['std']: {}, sys['std']: {}".format(sys_['std'], sys['std']))

    def test_ReturnsGeneratorDistributionFit_copy_obj_with_compute_change_scale_02(self):
        np.random.seed(0)
        loc, scale = 4, 10

        # Data creation  ---------------------------------------------
        # For dates references :
        folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data/Data_IMC_P01_D1")
        file_name = "USTEC.csv"

        load_file_args = dict(ref_colum_date="Time",
                              ref_colum_price="adj Close",
                              folder_path=folder_path)
        data_df = DataPreprocess.load_file_content(file_name=file_name, load_file_args=load_file_args)
        t_obj = st.t(loc=loc, scale=scale, df=4.5)
        data_df["USTEC"] = t_obj.rvs(size=data_df.shape[0])
        # ----------------------------------------------------------------

        distribution_fitter_params = dict(cost_function="MLE",  # "MLE",
                                          distribution_list=["Normal", "T"],  # ["Normal", "T"],
                                          use_fit_method=True)
        returns_generator_params_dict = dict(drop_out_of_date_median_values=False)
        rgdf = ReturnsGeneratorDistributionFit(data_df=data_df,
                                               returns_generator_params_dict=returns_generator_params_dict)
        rgdf.fit_distribution_to_data(distribution_fitter_params=distribution_fitter_params)

        sms, returns_samples = rgdf.compute_moments_from_sample(sample_size=50000)
        sds, cum_returns = rgdf.compute_daily_moments_from_sample(daily_sample_size=50000)
        sys, year_cum_returns = rgdf.compute_yearly_moments_from_sample(yearly_sample_size=10000)

        timedelta_value = 1
        rgdf_2 = rgdf.copy_obj_with_compute_change_scale(timedelta_unit="h", timedelta_value=timedelta_value)

        self.assertEqual(rgdf_2.frames_in_a_day, 24)
        self.assertEqual(rgdf_2.generate_daily_returns_samples_paths(2).shape, (2, 24))

        for i in range(len(rgdf.distribution_fitter.fitting_results)):
            loc_scaling = rgdf.distribution_fitter.fitting_results.iloc[i].parameters['loc']
            loc_scaling *= (rgdf.frames_in_a_day/rgdf_2.frames_in_a_day)

            scale_scaling = rgdf.distribution_fitter.fitting_results.iloc[i].parameters['scale']
            scale_scaling *= (rgdf.frames_in_a_day / rgdf_2.frames_in_a_day)**0.5

            self.assertEqual(rgdf_2.distribution_fitter.fitting_results.iloc[i].parameters['loc'],
                             loc_scaling)
            self.assertEqual(rgdf_2.distribution_fitter.fitting_results.iloc[i].distribution_obj_fitted.kwds['loc'],
                             loc_scaling)
            self.assertEqual(rgdf_2.distribution_fitter.fitting_results.iloc[i].parameters['scale'],
                             scale_scaling)
            self.assertEqual(rgdf_2.distribution_fitter.fitting_results.iloc[i].distribution_obj_fitted.kwds['scale'],
                             scale_scaling)

        sms_, returns_samples = rgdf_2.compute_moments_from_sample(sample_size=50000)
        sds_, cum_returns = rgdf_2.compute_daily_moments_from_sample(daily_sample_size=50000)
        sys_, year_cum_returns = rgdf_2.compute_yearly_moments_from_sample(yearly_sample_size=10000)

        self.assertAlmostEqual(sds_['E'], sds['E'], delta=0.5)
        self.assertAlmostEqual(sds_['std'], sds['std'], delta=0.5)

        self.assertAlmostEqual(sys_['E'], sys['E'], delta=10)
        self.assertAlmostEqual(sys_['std'], sys['std'], delta=10)

        print("sms_['E']: {}, sms['E']: {}".format(sms_['E'], sms['E']))
        print("sds_['E']: {}, sds['E']: {}".format(sds_['E'], sds['E']))
        print("sys_['E']: {}, sys['E']: {}".format(sys_['E'], sys['E']))

        print("sms_['std']: {}, sms['std']: {}".format(sms_['std'], sms['std']))
        print("sds_['std']: {}, sds['std']: {}".format(sds_['std'], sds['std']))
        print("sys_['std']: {}, sys['std']: {}".format(sys_['std'], sys['std']))

