from LinearRegression_ import LinearRegression_
import numpy as np


class MultipleLinearRegression(LinearRegression_):
    """
    Multiple Linear Regression Class, inherits LinearRegression_ class.
    """
    def fit_line(self, y_col, *argv):
        """
        Override self.fit_line(x_col, y_col); since we will have more than 1 x_col values.
        :param y_col: string; which indicates the attribute of the data frame, that user wish to place it into y-axis.
        :param argv: set of strings; which are indicates the attributes of the data frame, that user wish to place it into x-axis.
        :return:
        """
        y = self.df[y_col]
        if len(y.shape) == 1:
            y_ = y.values.reshape(-1, 1)
        else:
            y_ = y
        x_ = []
        for arg in argv:
            x_.append(self.df[arg])

        x_ = np.array(x_)
        x_ = x_.transpose()

        # x = self.df.iloc[:, [0, 2]].values
        # print(x == x_)
        self.linear_regression.fit(x_, y_)
        # print("b0: ", self.linear_regression.intercept_)
        # print("b1 to bn: ", self.linear_regression.coef_)
        # return self.linear_regression.intercept_, self.linear_regression.coef_

    def single_prediction(self, y_col, **kwargs):
        """
        Override self.single_prediction method!
        :param y_col: y_col: string; which indicates the attribute of the data frame, that user wish to place it into y-axis.
        :param kwargs: check for the main.py for the usage.
        :return:
        """
        keys, values = [], []
        for key, val in kwargs.items():
            keys.append(str(key))
            values.append(int(val))
        values = np.array([values])
        self.fit_line(y_col, *keys)
        return self.linear_regression.predict(values)

    def multiple_predictions(self, y_col, **kwargs):
        """
        Override self.multiple_predictions method!
        :param y_col: y_col: string; which indicates the attribute of the data frame, that user wish to place it into y-axis.
        :param kwargs: check for the main.py for the usage.
        :return:
        """
        keys, values = [], []
        for key, vals in kwargs.items():
            keys.append(str(key))
            values.append(vals)
        values = np.array(values).transpose()
        self.fit_line(y_col, *keys)
        return self.linear_regression.predict(values)