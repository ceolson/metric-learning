import numpy as np
from folktables import ACSDataSource, ACSMobility
from matplotlib import pyplot as plt
import torch
from scipy import stats
from scipy.sparse.linalg import lobpcg
import pandas as pd
from inFairness.distances import MahalanobisDistances, SquaredEuclideanDistance
from inFairness.fairalgo import SenSeI
from inFairness.auditor import SenSeIAuditor
from tqdm.auto import tqdm
from utils import *


if __name__ == '__main__':

    r = 3
    p = 30
    n = 200

    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["TX"], download=True)
    acs_data_adult = acs_data[acs_data["AGEP"] >= 18]

    acs_data_cleaned = acs_data_adult.select_dtypes(include=["float64", "int64"])


    acs_data_cleaned = acs_data_cleaned.loc[:, ~(acs_data_cleaned.isna().any())]
    acs_data_cleaned = acs_data_cleaned.loc[:, (acs_data_cleaned.var(axis=0) > 0)]
    acs_data_cleaned = acs_data_cleaned.sample(frac=1, axis=0)
    print(np.shape(np.array(acs_data_cleaned)))

    print("Generating synthetic data...")
    X = clean_data(2 * n, p, acs_data_cleaned)
    X_train = X[:n]
    X_test = X[n:]

    M, S, y, Astar, Kstar = generate_synthetic_data(n, r, p, X_train)

    print(np.shape(X_train), np.shape(X_test))

    np.save("Astar.npy", Astar)

    print("Initializing...")
    A0 = initialization(n, p, S, X_train, y)

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

    device = torch.device("cpu")


    K_hat = A.detach().cpu().numpy() @ A.detach().cpu().numpy().T

    learned_metric = MahalanobisDistances()
    learned_metric.fit(K_hat)
    learned_metric.to(device)

    true_metric = MahalanobisDistances()
    true_metric.fit(K)
    true_metric.to(device)

    output_metric = SquaredEuclideanDistance()
    output_metric.fit(num_dims=1)
    output_metric.to(device)

    Y = (acs_data_cleaned["PINCP"] > 25000).head(2 * n)
    Y_train = Y[:n]
    Y_test = Y[n:2 * n]

    class Model(torch.nn.Module):
        def __init__(self, input_size, output_size):

            super().__init__()
            self.fc1 = torch.nn.Linear(input_size, 100)
            self.fc2 = torch.nn.Linear(100, 100)
            self.fcout = torch.nn.Linear(100, output_size)

        def forward(self, x):

            x = torch.nn.functional.relu(self.fc1(x))
            x = torch.nn.functional.relu(self.fc2(x))
            x = self.fcout(x)
            return x

    X_train_tensor = torch.Tensor(X_train)
    X_test_tensor = torch.Tensor(X_test)

    Y_train_tensor = torch.Tensor(Y_train)
    Y_test_tensor = torch.Tensor(Y_test)

    rho = 5.0
    eps = 0.1
    auditor_nsteps = 100
    auditor_lr = 1e-3

    network = Model(np.shape(X_train)[1], 1).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    lossfn = torch.nn.functional.cross_entropy

    fairalgo = SenSeI(network, learned_metric, output_metric, lossfn, rho, eps, auditor_nsteps, auditor_lr)

    fairalgo.train()
    for epoch in tqdm(10):
        for x, y in zip(X_train_tensor, Y_test_tensor):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            result = fairalgo(x, y)
            result.loss.backward()
            optimizer.step()


