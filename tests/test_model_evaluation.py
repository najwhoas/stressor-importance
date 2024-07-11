import unittest
import pandas as pd
from src.model_evaluation import evaluate_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class TestModelEvaluation(unittest.TestCase):

    def setUp(self):
        self.data = pd.read_csv('./data/StressLevelDataset.csv')
        self.data['blood_pressure'] = self.data['blood_pressure'].map({1:0, 2:1, 3:2})

        # Assuming you want to clean the data as in the main code
        for var in self.data.columns:
            Q1 = self.data[var].quantile(0.25)
            Q3 = self.data[var].quantile(0.75)
            IQR = Q3 - Q1
            maximum = Q3 + (1.5 * IQR)
            minimum = Q1 - (1.5 * IQR)
            kondisi_lower_than = self.data[var] < minimum
            kondisi_more_than = self.data[var] > maximum
            self.data[var] = self.data[var].mask(cond=kondisi_more_than, other=maximum)
            self.data[var] = self.data[var].mask(cond=kondisi_lower_than, other=minimum)
        
        self.data = self.data.astype('int64')

        self.X = self.data.drop('stress_level', axis=1)
        self.y = self.data['stress_level']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.4, random_state=42)
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def test_evaluate_model(self):
        accuracy, precision, recall, f1, cm = evaluate_model(self.model, self.X_test, self.y_test)
        self.assertGreaterEqual(accuracy, 0.5)
        self.assertGreaterEqual(precision, 0.5)
        self.assertGreaterEqual(recall, 0.5)
        self.assertGreaterEqual(f1, 0.5)

if __name__ == '__main__':
    unittest.main()
