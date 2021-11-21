import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import LinearRegression_

        
def main():
    df = pd.read_csv("linear_regression_dataset.csv", sep=";")
    lr = LinearRegression_.LinearRegression_(df)
    # lr.visualize_data("deneyim", "maas")

    # print("Prediction on the second index: " ,lr.single_prediction("deneyim", "maas", 2))

    # temp = [30, 40, 60]
    # print(lr.multiple_predictions("deneyim", "maas", temp))

    # lr.visualize_results("deneyim", "maas", 50)
        
        
if __name__ == '__main__':
    main()