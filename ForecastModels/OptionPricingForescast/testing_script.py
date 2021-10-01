import os
import pandas as pd
import numpy as np
from scipy.optimize import fsolve

path_default = os.path.dirname(os.path.dirname(os.getcwd())) + '/data'
path_options_data = path_default + "/Options/"
file_name = "aapl-options-exp-2021-11-05-weekly-near-the-money-stacked-10-01-2021.csv"

aapl_options_exp_2021_11_05 = pd.read_csv(path_options_data + file_name)
calls_ = aapl_options_exp_2021_11_05[aapl_options_exp_2021_11_05["Type"] == "Call"].reset_index()
calls_["Moneyness_float"] = calls_["Moneyness"].str.replace("%", "").astype(float)
calls_["cp"] = calls_["Moneyness_float"]*calls_["Strike"]/100 + calls_["Strike"]

puts_ = aapl_options_exp_2021_11_05[aapl_options_exp_2021_11_05["Type"] == "Put"].reset_index()
puts_["Moneyness_float"] = puts_["Moneyness"].str.replace("%", "").astype(float)
puts_["cp"] = puts_["Strike"] - puts_["Moneyness_float"]*puts_["Strike"]/100


def get_call_put_prices(x_moneyness, calls, puts):
    def interpolate(x, x1, y1, x2, y2):
        return y1 + (x - x1) * (y2 - y1) / (x2 - x1)

    x_obj = x_moneyness
    for i in range(len(calls) - 1):
        if (calls.iloc[i]["Moneyness_float"] >= x_obj) and (calls.iloc[i + 1]["Moneyness_float"] < x_obj):
            x1_, y1_ = calls.iloc[i]["Moneyness_float"], calls.iloc[i]["Midpoint"]
            x2_, y2_ = calls.iloc[i + 1]["Moneyness_float"], calls.iloc[i + 1]["Midpoint"]
            p_obj_call = interpolate(x_obj, x1_, y1_, x2_, y2_)

    for i in range(len(puts) - 1):
        if (puts.iloc[i]["Moneyness_float"] < x_obj) and (puts.iloc[i + 1]["Moneyness_float"] >= x_obj):
            x1_, y1_ = puts.iloc[i]["Moneyness_float"], puts.iloc[i]["Midpoint"]
            x2_, y2_ = puts.iloc[i + 1]["Moneyness_float"], puts.iloc[i + 1]["Midpoint"]
            p_obj_puts = interpolate(x_obj, x1_, y1_, x2_, y2_)

    return p_obj_call, p_obj_puts


def get_call_put_moneyness(x_price, calls, puts):
    def interpolate(x, x1, y1, x2, y2):
        return y1 + (x - x1) * (y2 - y1) / (x2 - x1)

    x_obj = x_price
    for i in range(len(calls) - 1):
        if (calls.iloc[i]["Midpoint"] >= x_obj) and (calls.iloc[i + 1]["Midpoint"] < x_obj):
            y1_, x1_ = calls.iloc[i]["Moneyness_float"], calls.iloc[i]["Midpoint"]
            y2_, x2_ = calls.iloc[i + 1]["Moneyness_float"], calls.iloc[i + 1]["Midpoint"]
            moneyness_obj_call = interpolate(x_obj, x1_, y1_, x2_, y2_)

    for i in range(len(puts) - 1):
        if (puts.iloc[i]["Midpoint"] < x_obj) and (puts.iloc[i + 1]["Midpoint"] >= x_obj):
            y1_, x1_ = puts.iloc[i]["Moneyness_float"], puts.iloc[i]["Midpoint"]
            y2_, x2_ = puts.iloc[i + 1]["Moneyness_float"], puts.iloc[i + 1]["Midpoint"]
            moneyness_obj_puts = interpolate(x_obj, x1_, y1_, x2_, y2_)

    return moneyness_obj_call, moneyness_obj_puts


zero_moneyness_call_price, zero_moneyness_put_price = get_call_put_prices(0, calls=calls_, puts=puts_)
print("zero_moneyness_call_price:", zero_moneyness_call_price)
print("zero_moneyness_put_price:", zero_moneyness_put_price)

five_moneyness_call_price, five_moneyness_put_price = get_call_put_moneyness(5, calls=calls_, puts=puts_)
print("five_moneyness_call_price:", five_moneyness_call_price)
print("five_moneyness_put_price:", five_moneyness_put_price)


prices = []
for i in range(len(calls_) - 1):
    def for_solve(x):
        return [(x[0] - calls_.iloc[i]["Strike"]) / x[0] - calls_.iloc[i]["Moneyness_float"] / 100]
    prices.append(fsolve(for_solve, [150]))

S0 = np.mean(prices)

calls_["benefits"] = S0 - calls_["Strike"]
calls_["price - benefits"] = calls_["Midpoint"] - calls_["benefits"]

puts_["benefits"] = puts_["Strike"] - S0
puts_["price - benefits"] = puts_["Midpoint"] - puts_["benefits"]

# TODO_:
#   Generate several samples from normal using x (to solve) & market implied volatility
#   Compute option price & compare with current option price.
#   Solve for finding best x! (minimize difference!)
#   Idea:
#       Use also t distribution, (df fitted with historical stock values)?

# Actually sample from historical data with adjusted mean ?

# Model (for now just avoid interest rates):
#       C -> 5 ?
#   Prob(What happend)*(S - K) + C = Price
#   (1 - Prob(What happend)*(K - S) + C = Price

# For C -> use % -> 0 & interpolate ?

# Si ganar lo mismo cuesta mas para una Call significa que:
#       se espera que su probabilidad sea mas baja
# Si para un mismo cost. ie 5, se espera que ganes mas, es porque es menos probable ?


# Match
#   Given a % (probably needs to interpolate)
#   Match call with that Moneyness
#   Match put with that moneyness
#   Eq:
#       1) P * % + C = Call Price
#       2) (1 - P) * % + C = Put Price
#   Call
