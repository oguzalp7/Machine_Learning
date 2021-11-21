import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class LinearRegression_:
    """
    Simple Linear Regression
    y = b0 + b1 * x
    where:
        - y => dependent variable
        - b0 => intercept (constant or bias)
        - b1 => coefficient (slope of the fitted line)
        - x => independent variable
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

    def single_prediction(self, x_col, y_col, idx):
        """
        Single point prediction
        :param x_col: string; which indicates the attribute of the data frame, that user wish to place it into x-axis. 
        :param y_col: string; which indicates the attribute of the data frame, that user wish to place it into y-axis.
        :param idx: integer; the specified index for prediction. Encourage to set higher than total number of samples.
        """
        self.fit_line(x_col, y_col)
        return self.linear_regression.predict(np.array([[idx]]))

    def multiple_predictions(self, x_col, y_col, idx_array):
        """
        :param x_col: string; which indicates the attribute of the data frame, that user wish to place it into x-axis.
        :param y_col: string; which indicates the attribute of the data frame, that user wish to place it into y-axis.
        :param idx_array: list; containing the index numbers of the desired predictions.
        :return: nump array; containing the predictions based on the indexes of the list called "idx_array"
        """
        self.fit_line(x_col, y_col)
        try:
            idx_array = np.array(idx_array).reshape(-1, 1)
        except Exception as e:
            print("The array is already in a good shape. Additional information: ", e)
        return self.linear_regression.predict(idx_array)

    def visualize_results(self, x_col, y_col, num_of_samples, color="red"):
        """
        :param x_col: string; which indicates the attribute of the data frame, that user wish to place it into x-axis. 
        :param y_col: string; which indicates the attribute of the data frame, that user wish to place it into y-axis.
        :param num_of_samples: integer; indicates the number of desired predictions.
        :param color: string, indicates the color of the fitted line.
        :return: 
        """
        self.fit_line(x_col, y_col)
        temp_ = []
        for i in range(num_of_samples):
            temp_.append(i)
        array = np.array(temp_).reshape(-1, 1)
        y_head = self.linear_regression.predict(array)
        self.visualize_data(x_col=x_col, y_col=y_col, show=False)
        plt.plot(array, y_head, color=color)
        plt.show()
