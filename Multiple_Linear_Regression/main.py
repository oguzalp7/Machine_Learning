import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from LinearRegression_ import LinearRegression_
from MultipleLinearRegression import MultipleLinearRegression


def main():
    df = pd.read_csv("multiple_linear_regression_dataset.csv", sep=";")
    mlr = MultipleLinearRegression(df)
    # mlr.fit_line("salary", "experience", "age")
    # print(mlr.single_prediction("salary", experience=10, age=35))
    print(mlr.single_prediction("salary", experience=5, age=35))
    print(mlr.multiple_predictions("salary", experience=[5, 10, 15], age=[25, 30, 38]))


if __name__ == '__main__':
    main()
