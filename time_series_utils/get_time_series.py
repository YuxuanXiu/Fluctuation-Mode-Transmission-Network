import pandas as pd

def get_oil_price(data_path):
    df = pd.read_csv(data_path)
    time_series_dict = {}
    time_series_dict['timestamp_list'] = df['Date'].to_list()
    time_series_dict['idx_to_timestamp_dict'] = df['Date'].to_dict()
    time_series_dict['data_list'] = df['Price'].to_list()
    return time_series_dict

if __name__ == "__main__":
    data_path = "../data/oil_prices/brent-daily.csv"
    get_oil_price(data_path)

