# TODO_:
#   Fit several ARIMA model
#   Construct attribute with summary
#   Order selection of model according user criteria: #AIC, BIC, R_adjusted
#       Create an ENUM of different criterias
#       ( + others like -> all coefficients sig, ect.. (could be % of significant coefficients parameters))
#       Model criteria migth be optimized by styding forecasting errors
import os
import pandas as pd

from ForecastModels.ModelOne.ARIMASelCriteriaEnum import ARIMASelCriteriaEnum
from ForecastModels.ModelOne.ARIMAmodel import ARIMAmodel

max_ar, max_ma = 4, 2
default_order_list = [(0, 0, 0),
                      (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0), (5, 0, 0), (6, 0, 0),
                      (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4)]
default_order_list += [(i, 0, j) for i in range(1, max_ar + 1) for j in range(1, max_ma + 1)]


class AutoARIMAmodel(object):
    def __init__(self, data_df, order_list=default_order_list, selection_criteria=ARIMASelCriteriaEnum.BIC):
        self.models = None
        self.models_summary_df = None
        self.data_df = None
        self.order_list = order_list
        self.selection_criteria = None
        self.models_summary_criteria_df = None
        self.best_model = None

        self.run_models_estimations(data_df=data_df, order_list=order_list)
        self.run_models_summary_df()
        self.run_ordering_by_selection_criteria(selection_criteria)

    def run_models_estimations(self, data_df, order_list):
        self.data_df = data_df.copy()
        self.order_list = order_list.copy()
        self.models = dict()
        for order in order_list:
            model = ARIMAmodel(data_df=data_df, order=order, seasonal_order=(0, 0, 0, 0))
            self.models[model.name] = model

    def run_models_summary_df(self):
        self.models_summary_df = pd.DataFrame()
        for model in self.models.values():
            self.models_summary_df = self.models_summary_df.append(model.model_summary_series)
        self.models_summary_df = self.models_summary_df.astype({'has_significant_predictors_coefficients': bool,
                                                                'll_white_noise_residuals': bool,
                                                                'model_ok': bool,
                                                                'n_significant_predictors_coefficients': int,
                                                                'n_not_significant_predictors_coefficients': int})

    def run_ordering_by_selection_criteria(self, selection_criteria):
        self.selection_criteria = selection_criteria  # ARIMASelCriteriaEnum(selection_criteria)
        self.models_summary_df = self.models_summary_df.sort_values(by=[selection_criteria.model_summary_column],
                                                                    ascending=selection_criteria.ascending)

        self.models_summary_criteria_df = self.models_summary_df.copy()

        # check_model_ok:
        if selection_criteria.check_model_ok:
            self.models_summary_criteria_df = self.models_summary_criteria_df[self.models_summary_criteria_df.model_ok]

        # req_min_significant_predictors_percent:
        sig_coefficients_percent = self.models_summary_criteria_df.n_significant_predictors_coefficients \
                                   / (self.models_summary_criteria_df.n_significant_predictors_coefficients
                                      + self.models_summary_criteria_df.n_not_significant_predictors_coefficients)
        sub_select_models = sig_coefficients_percent >= selection_criteria.req_min_significant_predictors_percent
        self.models_summary_criteria_df = self.models_summary_criteria_df[sub_select_models]

        best_model_name = self.models_summary_criteria_df.iloc[0].name
        self.best_model = self.models[best_model_name]


if __name__ == '__main__':
    # Loadding of Data ===================================================================================================
    path_default = os.path.dirname(os.path.dirname(os.getcwd())) + '/data'
    path_module3_data = path_default + "/IMFx/Module3_data/"
    file_name = "module3_data_PE_Ratios.csv"

    module3_data_PE_Rations_df = pd.read_csv(path_module3_data + file_name)
    module3_data_PE_Rations_df = module3_data_PE_Rations_df.set_index(['dateid'])
    pe_saf_df = module3_data_PE_Rations_df['pe_saf'].dropna()
    # ====================================================================================================================

    auto_ARIMA_model = AutoARIMAmodel(data_df=pe_saf_df, selection_criteria=ARIMASelCriteriaEnum.R_squared_adj_spc)
