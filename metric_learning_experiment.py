import numpy as np
from folktables import ACSDataSource, ACSMobility
from matplotlib import pyplot as plt
import torch
from scipy import stats
from scipy.sparse.linalg import lobpcg
import pandas as pd
from utils import *

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
p = 30
n = 160

data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=["TX"], download=True)

results = pd.DataFrame(columns=[
        "Iterate",
        "AA^T to Kstar"
    ])

features, labels, _ = ACSMobility.df_to_pandas(acs_data)

X = clean_data(2 * n, len(features.columns), features, cut_columns=False)
X_train = X[:n]
X_test = X[n:]

Y = labels.head(2 * n)
Y_train = Y[:n]
Y_test = Y[n:]

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

for iterate in range(200 * n):
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

        results.to_csv("gd_out.csv")

Ahat = A_iterates[-1]

print("AA^T to Kstar:", np.linalg.norm(Ahat @ Ahat.T - Kstar))

results.to_csv("gd_out.csv")