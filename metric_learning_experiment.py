import numpy as np
from folktables import ACSDataSource, ACSMobility, ACSIncome
from matplotlib import pyplot as plt
import torch
from scipy import stats
from scipy.sparse.linalg import lobpcg
from scipy.linalg import eigh, eig
import pandas as pd
from inFairness.distances import MahalanobisDistances, SquaredEuclideanDistance, LogisticRegSensitiveSubspace
from inFairness.fairalgo import SenSeI
from inFairness.auditor import SenSeIAuditor, SenSRAuditor
from tqdm.auto import tqdm
from utils import *

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import data


r = 3
p = 20
n = 200

results = pd.DataFrame(columns=[
        "Iterate",
        "AA^T to Kstar"
    ])

chi = "ACS"
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=["TX"], download=True)
features, labels, _ = ACSMobility.df_to_pandas(acs_data)

# chi = 20
# synthetic_covariance_diag = chi * np.ones(p)
# synthetic_covariance_indices = np.random.choice(range(p), size=int(p / 2), replace=False)
# synthetic_covariance_diag[synthetic_covariance_indices] = 1 / chi
# features = pd.DataFrame(np.random.multivariate_normal(np.zeros(p), np.diag(synthetic_covariance_diag), size=n * 2))
# print("chi:", chi)

X = clean_data(2 * n, len(features.columns), features, cut_columns=False)
X_train = X[:n]
X_test = X[n:]

p = np.shape(X)[-1]

M, S, y, Astar, Kstar = generate_synthetic_data(n, r, p, X_train)

A0 = np.random.normal(0, 1, size=(p, r)) / (np.sqrt(p) * np.sqrt(r)) # initialization(n, r, p, S, X_train, y)
print(Astar)
print(A0)

A_iterates = []
A = torch.tensor(A0, requires_grad=True, device="cpu").to(torch.float64)
dists = []

y_tensor = torch.tensor(y, device="cpu")
M_tensor = torch.tensor(M, device="cpu")

for iterate in range(100 * n):
    loss = L(A, y_tensor, M_tensor)
    loss.backward()
    with torch.no_grad():
        A -= A.grad * 0.01
        A_iterates.append(A.detach().cpu().numpy())
        dists.append(np.linalg.norm(A.detach().cpu().numpy() @ A.detach().cpu().numpy().T - Kstar))
        A.grad.zero_()
    if iterate % 50 == -1 % 50:
        results.loc[len(results.index)] = {
            "Iterate": iterate,
            "AA^T to Kstar": dists[-1]
        }

        results.to_csv("gd_out_{}.csv".format(chi))

Ahat = A_iterates[-1]

print("AA^T to Kstar:", np.linalg.norm(Ahat @ Ahat.T - Kstar))

results.to_csv("gd_out_{}.csv".format(chi))
