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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(f"X_train:{X_train.shape}\ny_train:{y_train.shape}")
    w_init = np.repeat(0, X_train.shape[1]).reshape((X_train.shape[1],))
    algs = ['gd','bgd','acc','bac','nt','bnt']
    regressor = Regressor(w_init, learning_rate=0.1, check_stop=True)
    save_info(regressor, X_train, y_train, algs)

