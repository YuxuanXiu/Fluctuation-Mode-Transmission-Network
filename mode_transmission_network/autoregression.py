import statsmodels.api as sm
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.stattools import durbin_watson
from mode_transmission_network.dw_test import dwtest


def extract_autoregression_feature(this_data,this_timestamp):
    if type(this_data) is not np.ndarray:
        this_data = np.asarray(this_data)
    this_data = np.where(this_data != 0, np.log(this_data), 0)
    idx_nan = np.argwhere(np.isnan(this_data))
    this_data = np.delete(this_data, idx_nan)
    this_timestamp = np.delete(this_timestamp, idx_nan)
    reg_data = AutoReg(this_data, lags=1).fit(use_t=True)
    a = reg_data.params[0]
    b = reg_data.params[1]
    # test b with t test
    b_p_value = reg_data.pvalues[1]
    if b_p_value < 0.05:
        b_pass_t = True
    else:
        b_pass_t = False
    # test autocorrelation with DW test
    error = reg_data.resid
    p_value, dw = dwtest(error, this_data[:, np.newaxis], tail='both', method='Pan', matrix_inverse=False, n=15)
    if p_value < 0.01:
        e_pass_DW = True
    else:
        e_pass_DW = False
    if b_pass_t and e_pass_DW:
        return [a, b, "PASS FIRST TIME"]
    else:
        rho_bar = 1-dw/2
        a_star = a*(1-rho_bar)
        this_data_star = this_data - np.roll(this_data, 1,axis=0)*rho_bar
        this_data_star[0] = np.log(this_data[0]) + 0.5*np.log(1-rho_bar**2)
        error_star = this_data_star - a_star - b * np.roll(this_data_star, 1,axis=0)
        error_star = error_star[1:]
        p_value_star, dw_star = dwtest(error_star, this_data_star[:, np.newaxis], tail='both', method='Pan', matrix_inverse=False, n=15)
        if p_value_star < 0.01:
            e_pass_DW_star = True
        else:
            e_pass_DW_star = False
        b_star_rerun_pass_t = b_pass_t
        if b_star_rerun_pass_t and e_pass_DW_star:
            return [a_star, b, "PASS SECOND TIME"]
        elif b_star_rerun_pass_t and (not e_pass_DW_star):
            return [a_star, b, "D-NO DW"]
        elif (not b_star_rerun_pass_t) and e_pass_DW_star:
            return [a_star, b, "P-NO B SIG"]
        else:
            return [a_star, b, "PD-NEITHER"]






if __name__ == "__main__":
    data = sm.datasets.sunspots.load_pandas().data['SUNACTIVITY']
    out = 'AIC: {0:0.3f}, HQIC: {1:0.3f}, BIC: {2:0.3f}'
    print(extract_autoregression_feature(data.values, data.values))


