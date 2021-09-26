import os
import pandas as pd
import numpy as np


# Data Loading tools definitions =====================================================================================
path_default = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))) + '/data'
path_default_FRED = path_default + '/FRED'


def complete_load_data_df(file_df,
                          price_column_name,
                          price_column,
                          date_column,
                          date_threshold_min,
                          date_threshold_max,
                          time_dtype):
    to_rename_dict = dict()
    if price_column_name is not None:
        to_rename_dict[price_column] = price_column_name
    if date_column is not None:
        to_rename_dict[date_column] = "date_time"

    if to_rename_dict:
        file_df = file_df.rename(columns=to_rename_dict)
    file_df["date_time"] = pd.to_datetime(file_df["date_time"])

    if date_threshold_min is not None:
        file_df = file_df[file_df.date_time >= date_threshold_min].loc[:]
    if date_threshold_max is not None:
        file_df = file_df[file_df.date_time <= date_threshold_max].loc[:]

    file_df = file_df.set_index("date_time")

    if time_dtype is not None:
        dates_as_dtype = pd.Series(file_df.index).astype(time_dtype)
        file_df['dates_as_dtype'] = dates_as_dtype.values
        file_df = file_df.groupby('dates_as_dtype').first()

    return file_df


def load_FRED_data_df(file_name,
                      path,
                      sub_folder=None,
                      price_column=None,
                      date_column='DATE',
                      price_column_name=None,
                      date_threshold_min=None,
                      date_threshold_max=None):

    load_path_file = path
    if sub_folder is not None:
        load_path_file += "/" + sub_folder
    load_path_file += "/" + file_name

    file_df = pd.read_csv(load_path_file)

    if price_column is None:
        price_column = file_name.replace(".csv", "")

    return complete_load_data_df(file_df=file_df,
                                 price_column_name=price_column_name,
                                 price_column=price_column,
                                 date_column=date_column,
                                 date_threshold_min=date_threshold_min,
                                 date_threshold_max=date_threshold_max,
                                 time_dtype=None)


def read_book_data_file(file_name, path=path_default, delim_whitespace=True, header=None, columns_names=[], parse_date=True):
    file_df = pd.read_csv(os.path.join(path, file_name),
                          delim_whitespace=delim_whitespace,
                          header=header)
    file_df.columns = columns_names
    if parse_date:
        file_df['date'] = pd.to_datetime(file_df['date'], format="%Y%m%d")
        file_df = file_df.set_index('date')
    return file_df

# ====================================================================================================================
# Returns Processing tools definitions ===============================================================================


def compute_log_returns(price_df):
    return np.log(price_df).diff().dropna()


def compute_simple_returns(price_df):
    log_returns_df = compute_log_returns(price_df)
    return np.exp(log_returns_df) - 1

