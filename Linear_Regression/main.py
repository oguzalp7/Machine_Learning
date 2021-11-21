import pandas as pd
import LinearRegression_

        
def main():
    df = pd.read_csv("linear_regression_dataset.csv", sep=";")
    lr = LinearRegression_.LinearRegression_(df)
    lr.visualize_data("experience", "salary")

    print("Prediction on the second index: ", lr.single_prediction("experience", "salary", 2))

    indexes = [30, 40, 60]
    print("Predictions at 30th, 40th, 60th indexes", lr.multiple_predictions("experience", "salary", indexes))

    lr.visualize_results("experience", "salary", 50)
        
        
if __name__ == '__main__':
    main()