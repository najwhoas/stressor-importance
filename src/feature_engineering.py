def extract_features(data):
    X = data.drop('stress_level', axis=1)
    y = data['stress_level']
    return X, y
