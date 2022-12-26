import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    path = "out.csv"

    df = pd.read_csv(path, index_col='id')
    X, y = df.drop('ret', axis=1), df['ret']

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=101)

    regression = LinearRegression()
    regression.fit(Xtrain, ytrain)

    pred = regression.predict(Xtest)

    print(f"MSE: {mean_squared_error(ytest, pred)}")
