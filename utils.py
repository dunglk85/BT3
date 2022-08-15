import numpy as np
import pandas as pd

def load_data(file_name):
    dat = pd.read_csv(file_name)
    m = dat.shape[0]
    y = np.array(dat['price']).reshape((m,))
    col = np.ones((m, 1))
    X = dat.to_numpy(dat.drop(columns=['price'], inplace=True))
    X = np.concatenate((X, col), axis=1)
    y = y/1000000
    return X, y