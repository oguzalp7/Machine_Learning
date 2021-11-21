import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import LinearRegression_

        
def main():
    df = pd.read_csv("linear_regression_dataset.csv", sep=";")
    lr = LinearRegression_.LinearRegression_(df)
    lr.visualize_results("deneyim", "maas", 50)
        
        
if __name__ == '__main__':
    main()