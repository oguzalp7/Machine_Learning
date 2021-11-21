import pandas as pd
from DecisionTreeRegression import DecisionTreeRegression


def main():
    df = pd.read_csv("decision_tree_regression_dataset.csv", sep=";", header=None)
    dtr = DecisionTreeRegression()
    x = df.iloc[:, 0].values.reshape(-1, 1)
    y = df.iloc[:, 1].values.reshape(-1, 1)
    # dtr.visualize_data(x, y)
    # print(dtr.predict(x, y, 5.5))
    dtr.visualize_results(x, y, x_label="Ticket Level", y_label="Ticket Fee")


if __name__ == '__main__':
    main()