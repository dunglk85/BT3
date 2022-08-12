import numpy as np
import pandas as pd

def load_data(file_name):
    dat = pd.read_csv(file_name)
    m = dat.shape[0]
    y = np.array(dat['price']).reshape((m,))

    col = np.ones((m, 1))
    X = dat.to_numpy(dat.drop(columns=['price'], inplace=True))
    X = np.concatenate((X, col), axis=1)
    # X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    # y = (y - np.min(y, axis=0)) / (np.max(y, axis=0) - np.min(y, axis=0))
    return X, y