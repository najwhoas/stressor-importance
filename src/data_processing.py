from libs import *
import pandas as pd
def load_data(url):
    data = pd.read_csv(url)
    return data

def handle_outliers(data):
    for var in data.columns:
        Q1 = data[var].quantile(0.25)
        Q3 = data[var].quantile(0.75)
        IQR = Q3 - Q1
        maximum = Q3 + (1.5 * IQR)
        minimum = Q1 - (1.5 * IQR)
        kondisi_lower_than = data[var] < minimum
        kondisi_more_than = data[var] > maximum
        data[var] = data[var].mask(cond=kondisi_more_than, other=maximum)
        data[var] = data[var].mask(cond=kondisi_lower_than, other=minimum)
    data = data.astype('int64')
    return data