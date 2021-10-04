import unittest
import os
import numpy as np
import pandas as pd
import random

from SimulationScenarioEngine.DataPreprocesor.dataPreprocess import DataPreprocess


class TestDataPreprocessor(unittest.TestCase):

    def test_case_01(self):
        folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data/Data_IMC_P01_M10")

        file_names = ["AUS200.csv", "CHINA50.csv", "DE30.csv"]
        for file_name in file_names:
            load_file_args = dict(ref_colum_date="Time",
                                  ref_colum_price="adj Close",
                                  folder_path=folder_path)

            fc = DataPreprocess.load_file_content(file_name=file_name, load_file_args=load_file_args)

            pd_df_test = pd.read_csv(folder_path + "/" + file_name)

            for i in range(10):
                i_test = random.randint(1, len(pd_df_test))
                test_date_i1 = pd_df_test.iloc[i_test].Time
                test_ret = np.log(pd_df_test.iloc[i_test]["adj Close"]) - np.log(
                    pd_df_test.iloc[i_test - 1]["adj Close"])
                procesed_ret = fc[fc.columns[0]][test_date_i1]
                print("Ref: {}, \t date: {}, \t procesed_ret: {:.5f}, \t test_ret: {:.5f}".format(file_name,
                                                                                                  test_date_i1,
                                                                                                  procesed_ret,
                                                                                                  test_ret))
                self.assertAlmostEqual(procesed_ret, test_ret)

    def test_drop_out_of_date_median_values(self):
        folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data/Data_IMC_P01_M10")

        file_names = ["AUS200.csv", "CHINA50.csv", "DE30.csv"]
        for file_name in file_names:
            load_file_args = dict(ref_colum_date="Time",
                                  ref_colum_price="adj Close",
                                  folder_path=folder_path)

            fc = DataPreprocess.load_file_content(file_name=file_name, load_file_args=load_file_args)

            pd_df_test = DataPreprocess.drop_out_of_date_median_values(fc)

            for i in range(10):
                i_test = random.randint(1, len(pd_df_test))
                test_date_i1 = pd_df_test.index[i_test]
                test_ret = pd_df_test[pd_df_test.columns[0]][test_date_i1]
                procesed_ret = fc[fc.columns[0]][test_date_i1]
                print("Ref: {}, \t date: {}, \t procesed_ret: {:.5f}, \t test_ret: {:.5f}".format(file_name,
                                                                                                  test_date_i1,
                                                                                                  procesed_ret,
                                                                                                  test_ret))
                self.assertEqual(procesed_ret, test_ret)
