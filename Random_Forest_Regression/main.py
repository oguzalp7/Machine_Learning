import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DecisionTreeRegression import DecisionTreeRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


class RandomForestRegression(DecisionTreeRegression):
    def __init__(self, n_estimators, random_state):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.random_forest_regression = RandomForestRegressor(n_estimators=self.n_estimators,
                                                              random_state=self.random_state)

    def fit_model(self, x_col, y_col):
        self.random_forest_regression.fit(x_col, y_col)

    def predict(self, x_col, y_col, *args):
        args = np.array(args).reshape(-1, 1)
        self.fit_model(x_col, y_col)
        return self.random_forest_regression.predict(args)


def main():
    df = pd.read_csv("random_forest_regression_dataset.csv", sep=";", header=None)
    x = df.iloc[:, 0].values.reshape(-1, 1)
    y = df.iloc[:, 1].values.reshape(-1, 1)
    rfr = RandomForestRegression(n_estimators=100, random_state=42)
    rfr.visualize_results(x, y)
    print(rfr.predict(x, y, 7.8))


if __name__ == '__main__':
    main()