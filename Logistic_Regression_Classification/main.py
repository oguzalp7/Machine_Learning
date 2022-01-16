"""
Simple Neural Networks From Scratch
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json
import pickle


def json_to_dict(input_file):
    file = open(input_file, 'r')
    data = json.load(file)
    return data


def dict_to_pickle(dictionary):
    return pickle.dumps(dictionary)


def pickle_to_dict(pickled_msg):
    return pickle.loads(pickled_msg)


def dict_to_json(dictionary, outfile_name='parameters.json'):
    outfile = open(outfile_name, 'w')
    json.dump(dictionary, outfile)


class LogisticRegressionClassificationFromScratch:
    def __init__(self, dimension, bias=0.0, weights=None, learning_rate=0.01, iterations=150):
        self.dimension = dimension
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.bias = bias
        if weights is None:
            self.weights = np.full((dimension, 1), 0.01)
        else:
            self.weights = weights

    def logistic_regression(self, x_train, y_train, x_test, y_test):
        parameters, gradients, cost_list = self.update(x_train, y_train)
        y_prediction_test = self.predict(x_test)
        y_prediction_train = self.predict(x_train)

        # Print train/test Errors
        print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    def predict(self, x_test):
        z = self.sigmoid(np.dot(self.weights.T, x_test) + self.bias)
        y_pred = np.zeros((1, x_test.shape[1]))
        for i in range(z.shape[1]):
            if z[0, i] <= 0.5:
                y_pred[0, i] = 0
            y_pred[0, i] = 1
        return y_pred

    def update(self, x_train, y_train, save=False, visualize_results=True):
        index, cost_list1, cost_list2 = [], [], []
        ret_gradients = {}
        for i in range(self.iterations):
            cost, gradients = self.forward_backward_propagation(x_train, y_train)
            cost_list1.append(cost)
            ret_gradients = gradients
            self.weights = self.weights - self.learning_rate * gradients["derivative_weight"]
            self.bias = self.bias - self.learning_rate * gradients["derivative_bias"]
            if i % 10 == 0:
                cost_list2.append(cost)
                index.append(i)
                print(f"Cost after iteration #{i}: ", cost)
        parameters = {"weights": self.weights, "bias": self.bias}
        if save:
            dict_to_json(parameters)

        if visualize_results:
            plt.plot(index, cost_list2)
            plt.xticks(index, rotation='vertical')
            plt.xlabel("Number of Iterarion")
            plt.ylabel("Cost")
            plt.show()
        return parameters, ret_gradients, cost_list1

    def forward_backward_propagation(self, x_train, y_train):
        z = np.dot(self.weights.transpose(), x_train) + self.bias
        y_head = self.sigmoid(z)
        loss = - y_train * np.log(y_head) - 1 - (1 - y_train) * np.log(1 - y_head)
        cost = (np.sum(loss)) / x_train.shape[1]

        derivative_weight = (np.dot(x_train, (y_head - y_train).T)) / x_train.shape[
            1]  # x_train.shape[1]  is for scaling
        derivative_bias = np.sum(y_head - y_train) / x_train.shape[1]  # x_train.shape[1]  is for scaling
        gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
        return cost, gradients

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def train_test_split_(x, y, **kwargs):
        # test_size => proportion of the data separation
        # random_state => id of a random state. Just like the random seed or magic number for expert advisors.
        return train_test_split(x, y, **kwargs)  # returns x_train, x_test, y_train, y_test

    @staticmethod
    def transpose_test_train(*args):
        ret1, ret2 = [], ["x_train", "x_test", "y_train", "y_test"]
        ret = {}
        for arg in args:
            arg = np.array(arg)
            ret1.append(arg.T)
        for i in range(4):
            ret[ret2[i]] = ret1[i]
        return ret


def main():
    # pre-process data
    df = pd.read_csv("data.csv")
    df.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
    df["diagnosis"] = [1 if each == "M" else 0 for each in df["diagnosis"]]
    y = df["diagnosis"]
    x_data = df.drop(["diagnosis"], axis=1)
    x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

    lrc = LogisticRegressionClassificationFromScratch(x.shape[1], iterations=150)
    test_train_set = (lrc.train_test_split_(x, y, test_size=0.2, random_state=42))
    input_sets = lrc.transpose_test_train(*test_train_set)  # x_train, x_test, y_train, y_test
    lrc.logistic_regression(input_sets["x_train"], input_sets["y_train"], input_sets["x_test"], input_sets["y_test"])


if __name__ == '__main__':
    main()
