import numpy as np
from folktables import ACSDataSource, ACSMobility
from matplotlib import pyplot as plt
import torch
from scipy import stats
from scipy.sparse.linalg import lobpcg
import pandas as pd
from utils import *

if __name__ == '__main__':

    synthetic_data = pd.DataFrame(np.random.normal(size=(4000, 50)))

    r = 3
    p = 10

    for n in range(1000, 3000, 100): # [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]:
        
        X = clean_data(n, p, synthetic_data)
        M, S, y, Astar, Kstar = generate_synthetic_data(n, r, p, X)
        
        A0 = initialization(n, r, p, S, X, y)
        
        print(n)
        print(np.linalg.norm(A0 @ A0.T - Kstar))
