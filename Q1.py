import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def main():
    Data = pd.read_csv("Advertising.csv")

    print("Dataset Loaded Successfully")
    print(Data.head())

    X = Data[['TV', 'radio', 'newspaper']]
    Y = Data['sales']

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.5, random_state=42
    )

    Model = LinearRegression()

    Model.fit(X_train, Y_train)

    print("Model Training Completed")

    Y_pred = Model.predict(X_test)

    print("\nPredicted Sales:")
    print(Y_pred)

    print("\nExpected Sales:")
    print(Y_test.values)

if __name__ == "__main__":
    main()