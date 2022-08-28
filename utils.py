import numpy
import numpy as np
import pandas as pd
import time
import math

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

def save_data(reg, X,y,w_init, algs):
    d = {}
    for a in algs:
        d[a] = reg.fit(X, y, w_init, a)
    df = pd.DataFrame(data=d)
    file_name = f'Ouput/cost_lr_{reg.lr}.csv'
    df.to_csv(file_name, index=False)

def save_info(reg, X,y,w_init, algs):
    df = pd.DataFrame()
    for a in algs:
        start = time.time()
        costs = reg.fit(X,y,w_init,a)
        end = time.time()
        score = reg.score(X,y)
        col = [costs[-1], math.sqrt(reg.square_norm), end - start, len(costs), reg.inner_count,score]
        df[a] = col
    file_name = f'Ouput/info_lr_{reg.lr}.csv'
    df.to_csv(file_name, index=False)

def save_cost_al(reg, X,y, a):
    costs = reg.fit(X, y, a)
    df = pd.DataFrame(data=costs)
    file_name = f'output_cost_lr_{reg.lr}_{a}.csv'
    df.to_csv(file_name, index=False)
def save_info_al(reg, X,y, a):
    start = time.time()
    costs = reg.fit(X, y, a)
    end = time.time()
    info = [costs[-1], math.sqrt(reg.square_norm), end - start, len(costs), reg.inner_count]
    df = pd.DataFrame(data=info)
    file_name = f'info_lr_{reg.lr}_{reg.tol}_{a}.csv'
    df.to_csv(file_name, index=False)
