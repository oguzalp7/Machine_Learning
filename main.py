from Linear_Regression.LinearRegression_ import LinearRegression_
from Multiple_Linear_Regression.MultipleLinearRegression import MultipleLinearRegression
from Polynomial_Linear_Regression.PolynomialLinearRegression import PolynomialLinearRegression
import pandas as pd


def main():
    lr_df = pd.read_csv("linear_regression_dataset.csv", sep=";")  # linear regression data frame
    mlr_df = pd.read_csv("multiple_linear_regression_dataset.csv", sep=";")  # multiple linear regression data frame
    plr_df = pd.read_csv("polynomial_regression.csv", sep=";")
    lr = LinearRegression_.LinearRegression_(lr_df)
    mlr = MultipleLinearRegression(mlr_df)
    plr = PolynomialLinearRegression(plr_df, 4)

    lr.visualize_results("experience", "salary", 50)
    print(mlr.multiple_predictions("salary", experience=[5, 10, 15], age=[25, 35, 38]))
    print(plr.single_prediction("price", "max_speed", 220))


if __name__ == '__main__':
    main()