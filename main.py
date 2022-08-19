from sklearn.model_selection import train_test_split
from utils import *
from linear_regression import *
import time

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    X, y = load_data('kc_house_data_cleaned.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(f"X_train:{X_train.shape}\ny_train:{y_train.shape}")
    w_init = np.random.random(X_train.shape[1])
    regressor = Regressor(w_init)

    algs=['gd','bgd','acc','bac','nt','bnt']
    algs = ['nt','bnt']
    min = 0
    max = 1
    for a in algs:
        start = time.time()
        costs = regressor.fit(X_train,y_train,a)
        end = time.time()
        label=("Algorithm"+ a)
        plt.plot(range(len(costs)), costs, label=a)
    plt.legend(algs)
    plt.show()
