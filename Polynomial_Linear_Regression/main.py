import pandas as pd
from PolynomialLinearRegression import PolynomialLinearRegression


def main():
    df = pd.read_csv("polynomial_regression.csv", sep=";")
    p = PolynomialLinearRegression(df, 4)
    # p.visualize_data("price", "max_speed")
    p.visualize("price", "max_speed")
    print(p.single_prediction("price", "max_speed", 220))
    
    temp = [150, 200, 220, 230, 180]
    print(p.multiple_predictions("price", "max_speed", temp))


if __name__ == '__main__':
    main()
