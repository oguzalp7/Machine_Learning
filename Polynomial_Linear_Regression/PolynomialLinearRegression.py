import numpy as np
from LinearRegression_ import LinearRegression_
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt


class PolynomialLinearRegression(LinearRegression_):
    """
    y = b0 + b1 * x + b2 * x^2 + b3 * x^3 + ... + bn * x^n
    where:
        - y => dependent variable
        - b0 => intercept (constant or bias)
        - b1 to bn => coefficients per sample, this is also the {Linear} part of the regression technique.
        - x => independent variable, this is also the {Polynomial} part of the regression technique.
    """
    def __init__(self, df, degree):
        self.df = df
        self.degree = degree
        self.polynomial_linear_regression = PolynomialFeatures(degree=self.degree)
        self.linear_regression = LinearRegression_(self.df).linear_regression

    def fit_line(self, x_col, y_col):
        """
        Override Parent Method
        :param x_col:
        :param y_col:
        :return:
        """
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
        x_polynomials = self.polynomial_linear_regression.fit_transform(x_)
        self.linear_regression.fit(x_polynomials, y_)
        return self.linear_regression.predict(x_polynomials), x_, y_

    def visualize(self, x_col, y_col):
        y_head, x_, y_ = self.fit_line(x_col, y_col)
        self.visualize_data(x_col, y_col, show=False)
        plt.plot(x_, y_head, color="green")
        plt.legend()
        plt.show()

    def single_prediction(self, x_col, y_col, idx):
        """
        Override Parent Method
        :param x_col: string; which indicates the attribute of the data frame, that user wish to place it into x-axis.
        :param y_col: string; which indicates the attribute of the data frame, that user wish to place it into y-axis.
        :param idx: integer; the specified index for prediction.  Encourage to set higher than total number of samples.
        """
        _, _, _ = self.fit_line(x_col, y_col)
        index = np.array([[idx]])
        return self.linear_regression.predict(self.polynomial_linear_regression.fit_transform(index))

    def multiple_predictions(self, x_col, y_col, idx_array):
        """
        Override Parent Method
        :param x_col: string; which indicates the attribute of the data frame, that user wish to place it into x-axis.
        :param y_col: string; which indicates the attribute of the data frame, that user wish to place it into y-axis.
        :param idx_array: list; containing the index numbers of the desired predictions.
        :return: nump array; containing the predictions based on the indexes of the list called "idx_array"
        """
        _, _, _ = self.fit_line(x_col, y_col)
        try:
            idx_array = np.array(idx_array).reshape(-1, 1)
        except Exception as e:
            print("The array is already in a good shape. Additional information: ", e)
        idx_array = self.polynomial_linear_regression.fit_transform(idx_array)
        return self.linear_regression.predict(idx_array)