from src.libs import *
from src.data_processing import *
from src.feature_engineering import *

url = './data/StressLevelDataset.csv'
data = load_data(url)

blood_pressure_mapping = {1: 0, 2: 1, 3: 2}
data = map_categorical(data, 'blood_pressure', blood_pressure_mapping)
data = handle_outliers(data)