import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    data['blood_pressure'] = data['blood_pressure'].map({1:0, 2:1, 3:2})
    for var in data.columns:
        Q1 = data[var].quantile(0.25)
        Q3 = data[var].quantile(0.75)
        IQR = Q3 - Q1
        maximum = Q3 + (1.5 * IQR)
        minimum = Q1 - (1.5 * IQR)
        data[var] = data[var].mask(data[var] > maximum, maximum)
        data[var] = data[var].mask(data[var] < minimum, minimum)
    data = data.astype('int64')
    return data
