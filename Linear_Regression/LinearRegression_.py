import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class LinearRegression_:
    """
    Simple Linear Regression
    """
    def __init__(self, data_frame):
        self.df = data_frame
        self.linear_regression = LinearRegression()

    def visualize_data(self, x_col, y_col, show=True):
        """
        x_col and y_col are the attributes of the data frame. In this function,
        they have to pass as string parameters.
        """
        plt.scatter(self.df[x_col], self.df[y_col])
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        if show:
            plt.show()

    def fit_line(self, x_col, y_col):
        x = self.df[x_col]
        y = self.df[y_col]
        if len(x.shape) == 1:
            x_ = x.values.reshape(-1, 1)
        else:
            x_ = x
        if len(y.shape) == 1:
            y_ = y.values.reshape(-1, 1)
        else:
            y_ = y
        self.linear_regression.fit(x_, y_)
        # print(x_.shape, y_.shape)
        # b0 = self.linear_regression.intercept_
        # b1 = self.linear_regression.coef_
        # print(self.linear_regression.predict(np.array([[11]])))

    def predict(self, x_col, y_col, idx):
        """
        x_col and y_col are strings which are related to the attributes of the data frame.
        idx parameter is an integer for the index of the x-axis.
        """
        self.fit_line(x_col, y_col)
        print(self.linear_regression.predict(np.array([[idx]])))

    def visualize_results(self, x_col, y_col, num_of_samples, color="red"):
        self.fit_line(x_col, y_col)
        temp_ = []
        for i in range(num_of_samples):
            temp_.append(i)
        array = np.array(temp_).reshape(-1, 1)
        y_head = self.linear_regression.predict(array)
        self.visualize_data(x_col=x_col, y_col=y_col, show=False)
        plt.plot(array, y_head, color=color)
        plt.show()
