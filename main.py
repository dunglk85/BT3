from sklearn.model_selection import train_test_split
from utils import *
from linear_regression import *
import random


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    X, y = load_data('kc_house_data_cleaned.csv')
    random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(f"X_train:{X_train.shape}\ny_train:{y_train.shape}")
    w_init = np.repeat(0, X_train.shape[1]).reshape((X_train.shape[1],))
    algs = ['gd','bgd','acc','bac','nt','bnt']
    lrs = [0.2, 0.1, 0.05, 0.02]
    for l in lrs:
        regressor = Regressor(learning_rate=l, check_stop=False, tol=1e-4, max_iters=10000)
        save_data(regressor, X_train, y_train, w_init, algs)
        regressor = Regressor(learning_rate=l, check_stop=True, tol=1e-4)
        save_info(regressor, X_train, y_train, w_init, algs)

