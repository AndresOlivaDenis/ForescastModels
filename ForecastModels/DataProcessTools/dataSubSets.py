def create_sub_samples(series_df, size):
    series_df_dict = dict()
    N = len(series_df)
    for i in range(size, N):
        date = series_df.iloc[i:i + 1].index.values[0]
        values = series_df.iloc[i-size:i+1]
        series_df_dict[date] = values
    return series_df_dict


def create_not_match_sub_samples(series_df, size):
    series_df_dict = dict()
    N = len(series_df)
    for i in range(size, N, size + 1):
        date = series_df.iloc[i:i + 1].index.values[0]
        values = series_df.iloc[i - size:i + 1]
        series_df_dict[date] = values
    return series_df_dict


def create_yearly_sub_samples(series_df):
    # _TODO
    pass


def create_yearly_month_sub_samples(series_df):
    # _TODO
    pass