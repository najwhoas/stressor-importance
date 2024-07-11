from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, n_estimators=75, max_depth=2, min_samples_split=2, max_features=0.075, random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, criterion='gini', max_features=max_features, random_state=random_state)
    model.fit(X_train, y_train)
    return model
