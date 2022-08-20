import numpy as np
import pandas as pd

def load_data(file_name):
    dat = pd.read_csv(file_name)
    m = dat.shape[0]
    y = np.array(dat['price']).reshape((m,))
    col = np.ones((m, 1))
    X = dat.to_numpy(dat.drop(columns=['price'], inplace=True))
    print(X.shape)
    X = np.concatenate((X, col), axis=1)
    y = y/1e6
    return X, y

def save_data(reg, X,y):
    algs = ['bg','bgd','acc','bac','nt','bnt']
    d = {}
    for a in algs:
        d[a] = reg.fit(X,y,a)
    df = pd.DataFrame(data=d)
    file_name = 'output_cost4.csv'
    df.to_csv(file_name, index=False)

