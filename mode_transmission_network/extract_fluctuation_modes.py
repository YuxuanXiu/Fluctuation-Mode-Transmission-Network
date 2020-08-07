import numpy as np
from mode_transmission_network.autoregression import extract_autoregression_feature

def get_all_fluctuation_modes(time_series_dict, window_length):
    timestamp_list = time_series_dict['timestamp_list']
    idx_to_timestamp_dict = time_series_dict['idx_to_timestamp_dict']
    data_list = time_series_dict['data_list']
    time_series_length = len(timestamp_list)
    fluctuation_mode_list = []
    # 0,1,2,...,
    for i in range(time_series_length - window_length):
        this_data = data_list[i: i + window_length]
        this_timestamp = timestamp_list[i: i + window_length]
        this_data_np_array = np.asarray(this_data)
        print(extract_autoregression_feature(this_data_np_array, this_timestamp))



