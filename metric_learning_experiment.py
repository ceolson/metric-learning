import numpy as np
from folktables import ACSDataSource, ACSMobility
from matplotlib import pyplot as plt
import torch
from scipy import stats
from scipy.sparse.linalg import lobpcg
import pandas as pd
from utils import *

if __name__ == '__main__':

    # synthetic_data = pd.DataFrame(np.random.normal(size=(1000, 50)))
    synthetic_data = pd.DataFrame(np.random.uniform(low=-1, high=1, size=(1000, 50)))

    r = 3
    p = 30
    n = 400


    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["AL"], download=True)

    acs_data_cleaned = acs_data.select_dtypes(include="float64")
    acs_data_cleaned = acs_data_cleaned.loc[:, ~(acs_data_cleaned.isna().mean(axis=0) > 0.78)]
    acs_data_cleaned = acs_data_cleaned.loc[~acs_data_cleaned.isna().any(axis=1)]
    acs_data_cleaned = acs_data_cleaned.loc[:, (acs_data_cleaned.var(axis=0) > 0)]
    acs_data_cleaned = acs_data_cleaned.sample(frac=1, axis=0)
    print(np.shape(np.array(acs_data_cleaned)))


    print("Generating synthetic data...")
    X = clean_data(n, p, synthetic_data)
    M, S, y, Astar, Kstar = generate_synthetic_data(n, r, p, X)
    print(np.shape(X))

    np.save("Astar.npy", Astar)

    print("Initializing...")
    A0 = initialization(n, p, S, X, y)

    print(np.linalg.norm(Astar - A0))



    print("Starting gradient descent...")

    A_iterates = []
    A = torch.tensor(A0, requires_grad=True, device="cuda")
    dists = []

    y_tensor = torch.tensor(y, device="cuda")
    M_tensor = torch.tensor(M, device="cuda")
    for iterate in range(2000):
        loss = L(A, y_tensor, M_tensor)
        loss.backward()
        with torch.no_grad():
            A -= A.grad * 0.1
            A_iterates.append(A.detach().cpu().numpy())
            dists.append(np.linalg.norm(A.detach().cpu().numpy() @ A.detach().cpu().numpy().T - Kstar))
            A.grad.zero_()
        if iterate % 10 == 9:
            print(iterate + 1, loss, np.linalg.norm(A.detach().cpu().numpy() @ A.detach().cpu().numpy().T - Kstar))

    np.save("A_iterates.npy", A_iterates)

    plt.plot(dists)
    plt.xlabel("iteration")
    plt.ylabel("||AA^T - K*||")

    plt.savefig("gd.png")
