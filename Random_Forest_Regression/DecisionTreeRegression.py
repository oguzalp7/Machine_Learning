import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


class DecisionTreeRegression:
    def __init__(self):
        self.tree_regressor = DecisionTreeRegressor()

    @staticmethod
    def visualize_raw_data(x_col, y_col, x_label=None, y_label=None, show=True, color="red"):
        plt.scatter(x_col, y_col, color=color)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if show:
            plt.show()

    def fit_model(self, x_col, y_col):
        self.tree_regressor.fit(x_col, y_col)

    def predict(self, x_col, y_col, *args):
        args = np.array(args).reshape(-1, 1)
        self.fit_model(x_col, y_col)
        return self.tree_regressor.predict(args)

    def scale_data(self, x_col, scale_rate=0.01):
        return np.arange(min(x_col), max(x_col), scale_rate).reshape(-1, 1)

    def visualize_results(self, x_col, y_col, x_label=None, y_label=None, color="green"):
        self.visualize_raw_data(x_col, y_col, show=False)
        x_ = self.scale_data(x_col)
        y_head = self.predict(x_col, y_col, *x_)
        plt.plot(x_, y_head, color=color)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()