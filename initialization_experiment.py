import numpy as np
from folktables import ACSDataSource, ACSMobility
from matplotlib import pyplot as plt
import torch
from scipy import stats
from scipy.sparse.linalg import lobpcg
import pandas as pd

if __name__ == '__main__':

    synthetic_data = pd.DataFrame(np.random.normal(size=(4000, 50)))

    r = 3
    p = 10

    for n in range(100, 3000, 5): # [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]:
        
        print("Generating synthetic data...")
        X = clean_data(n, p, synthetic_data)
        S, y, Astar, Kstar = generate_synthetic_data(n, r, p, X)
        print(np.shape(X))
        
        np.save("Astar.npy", Astar)
        
        print("Initializing...")
        A0 = initialization(n, p, S, X, y)
        
        print(n)
        print(np.linalg.norm(Astar - A0))
