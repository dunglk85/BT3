from sklearn.model_selection import train_test_split
from utils import *
from linear_regression import *

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    X, y = load_data('kc_house_data_cleaned.csv')
    # split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # print train and test set shapes
    print(f"X_train:{X_train.shape}\ny_train:{y_train.shape}")

    regressor = Regressor()
    # call the fit method
    regressor.fit(X_train, y_train)

    train_score = regressor.score(X_train, y_train)
    test_score = regressor.score(X_test, y_test)

    print("Train Score:", train_score)
    print("Test Score: ", test_score)
    regressor.plot()
