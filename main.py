from mode_transmission_network.extract_fluctuation_modes import get_all_fluctuation_modes
from time_series_utils.get_time_series import get_oil_price

if __name__ == "__main__":
    data_path = "data/oil_prices/brent-daily.csv"
    ts_dict = get_oil_price(data_path)
    get_all_fluctuation_modes(ts_dict,50)