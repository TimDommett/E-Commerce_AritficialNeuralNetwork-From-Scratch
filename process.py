# Preprocessing E-Commerce data

import numpy as np
import pandas as pd

# Look at data
# df = pd.read_csv('ecommerce_data.csv')
# print(df.head())

def get_data():
    df = pd.read_csv('ecommerce_data.csv')
    # transform data into matrix
    data = df.as_matrix()

    X = data[:, :-1]
    Y = data[:, -1]

    # normalise numerical columns:
    X[:,1] = (X[:,1] - X[:,1].mean()) / X[:, 1].std()
    X[:,2] = (X[:,2] - X[:,2].mean()) / X[:, 2].std()

    # Onehot encoding categorical columns
    N, D = X.shape
    X2 = np.zeros((N, D+3)) # +3 as there are 4 different categorical values
    X2[:,0:(D-1)] = X[:,0:(D-1)]

    for n in range(N):
        # time of day
        t = int(X[n,D-1])
        X2[n,t+D-1] = 1
    Z = np.zeros((N, 4))
    Z[np.arange(N), X[:,D-1].astype(np.int32)] = 1
    # X2[:,-4:] = z
    assert(np.abs(X2[:,-4:] - Z).sum() < 10e-10)

    return X2, Y

def get_binary_data():
    X, Y = get_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]
    return X2, Y2