
from enum import Enum


class ARIMASelCriteriaEnum(Enum):

    def __new__(cls, name, model_summary_column, ascending, check_model_ok, req_min_significant_predictors_percent):
        object_new = object.__new__(cls)
        object_new._value_ = name
        object_new.model_summary_column = model_summary_column
        object_new.ascending = ascending
        object_new.check_model_ok = check_model_ok
        object_new.req_min_significant_predictors_percent = req_min_significant_predictors_percent
        return object_new

    AIC = ('AIC', 'AIC', True, True, 0)
    BIC = ('BIC', 'BIC', True, True, 0)
    R_squared_adj = ('R_squared_adj', 'R_squared_adj', False, True, 0)

    AIC_spc = ('AIC_spc', 'AIC', True, True, 2.0/3.0)
    BIC_spc = ('BIC_spc', 'BIC', True, True, 2.0/3.0)
    R_squared_adj_spc = ('R_squared_adj_spc', 'R_squared_adj', False, True, 2.0/3.0)

    AIC_aspc = ('AIC_aspc', 'AIC', True, True, 1.0)
    BIC_aspc = ('BIC_aspc', 'BIC', True, True, 1.0)
    R_squared_adj_aspc = ('R_squared_adj_aspc', 'R_squared_adj', False, True, 1.0)
