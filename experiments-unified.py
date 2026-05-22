import numpy as np
from folktables import ACSDataSource, ACSMobility, ACSEmployment
from matplotlib import pyplot as plt
import torch
from scipy import stats
from scipy.sparse.linalg import lobpcg
from scipy.linalg import eigh, eig
from sklearn.decomposition import PCA
import pandas as pd
from inFairness.distances import MahalanobisDistances, SquaredEuclideanDistance, LogisticRegSensitiveSubspace
from inFairness.fairalgo import SenSeI
from inFairness.auditor import SenSeIAuditor, SenSRAuditor
from tqdm.auto import tqdm
from utils import *

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import data

from ucimlrepo import fetch_ucirepo
import sys

r = 3
p = 20
n = 120

dataset = sys.argv[1]
T = int(sys.argv[2])

if dataset.split(".")[0] == "gaussian":
    chi = float(dataset.split(".")[1])
    synthetic_covariance_diag = chi * np.ones(p)
    synthetic_covariance_indices = np.random.choice(range(p), size=int(p / 2), replace=False)
    synthetic_covariance_diag[synthetic_covariance_indices] = 1 / chi
    features = pd.DataFrame(np.random.multivariate_normal(np.zeros(p), np.diag(synthetic_covariance_diag), size=n * 2))

    signal = np.random.normal(0, 1, size=p)
    probs = 1 / (1 + np.exp(np.array(features) @ signal))
    labels = (np.random.uniform(size=2*n) < probs)
    labels = pd.DataFrame(labels).astype(int)

elif dataset.split(".")[0] == "binomial":
    chi = float(dataset.split(".")[1])
    probs = 1/chi * np.ones(p)
    probs_indices = np.random.choice(range(p), size=int(p / 2), replace=False)
    probs[probs_indices] = 1 - 1/chi
    features = pd.DataFrame(np.random.binomial(1, probs, size=(n * 2, p)))

    signal = np.random.normal(0, 1, size=p)
    probs = 1 / (1 + np.exp(np.array(features) @ signal))
    labels = (np.random.uniform(size=2*n) < probs)
    labels = pd.DataFrame(labels).astype(int)

elif dataset.split(".")[0] == "ar":
    chi = float(dataset.split(".")[1])
    rho = 1 - 1 / chi
    synthetic_covariance = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            synthetic_covariance[i, j] = rho ** np.abs(i - j)
    features = pd.DataFrame(np.random.multivariate_normal(np.zeros(p), synthetic_covariance, size=n * 2))

    signal = np.random.normal(0, 1, size=p)
    probs = 1 / (1 + np.exp(np.array(features) @ signal))
    labels = (np.random.uniform(size=2*n) < probs)
    labels = pd.DataFrame(labels).astype(int)

elif dataset == "ACSEmployment":
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["TX"], download=True)
    features, labels, _ = ACSEmployment.df_to_pandas(acs_data)
    for column in ["MAR", "ESP", "MIG", "CIT", "MIL", "ANC", "RAC1P", "RELP"]:
        one_hot = pd.get_dummies(features[column])
        for c in one_hot.columns:
            one_hot = one_hot.rename(columns={c: f"{column}.{int(c)}"})
        features = features.join(one_hot)
        features = features.drop(column, axis=1)

elif dataset == "ACSMobility":
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["TX"], download=True)
    features, labels, _ = ACSMobility.df_to_pandas(acs_data)
    for column in ["MAR", "ESP", "CIT", "MIL", "ANC", "RAC1P", "COW", "ESR", "RELP"]:
        one_hot = pd.get_dummies(features[column])
        for c in one_hot.columns:
            one_hot = one_hot.rename(columns={c: f"{column}.{int(c)}"})
        features = features.join(one_hot)
        features = features.drop(column, axis=1)

elif dataset == "CreditCardDefault":
    default_of_credit_card_clients = fetch_ucirepo(id=350)
    features = default_of_credit_card_clients.data.features
    labels = default_of_credit_card_clients.data.targets

    features["Married_Indicator"] = (features["X4"] == 1)
    features = features.drop("X4", axis=1)

elif dataset == "CommunityCrime":
    communities_and_crime = fetch_ucirepo(id=183)
    X = communities_and_crime.data.features
    features = X[np.setdiff1d(X.columns, ["state", "county", "community", "communityname", "fold"])]
    not_missing = ~(features == "?").any(axis=1)
    features = features[not_missing]
    features = features.sample(p, axis=1)
    labels = communities_and_crime.data.targets[not_missing]

elif dataset == "CDCDiabetes":
    cdc_diabetes_health_indicators = fetch_ucirepo(id=891)
    features = cdc_diabetes_health_indicators.data.features
    labels = cdc_diabetes_health_indicators.data.targets

X = clean_data(2 * n, p, features, cut_columns=False, pca=True)
X_train = X[:n]
X_test = X[n:]

Y = labels.head(2 * n)
Y_train = Y[:n]
Y_test = Y[n:]

p = X_train.shape[1]

M, S, y, Astar, Kstar = generate_synthetic_data(n, r, p, X_train)

with open('numpy_saves_{}.npz'.format(dataset), 'wb') as f:
    np.savez(f,
             S=S,
             y=y,
             Astar=Astar,
             Kstar=Kstar,
             X_train=X_train,
             X_test=X_test,
             Y_train=Y_train,
             Y_test=Y_test
            )

A0 = initialization(n, r, p, S, X_train, y)

A_iterates = []
A = torch.tensor(A0, requires_grad=True, device="cpu").to(torch.float64)
dists = []

y_tensor = torch.tensor(y, device="cpu")
M_tensor = torch.tensor(M, device="cpu")

for iterate in range(T): # 20 * n):
    loss = L(A, y_tensor, M_tensor)
    loss.backward()
    with torch.no_grad():
        A -= A.grad * 0.1
        print(A.detach().cpu().numpy())
        A_iterates.append(np.array(A.detach().cpu().numpy()))
        dists.append(np.linalg.norm(A.detach().cpu().numpy() @ A.detach().cpu().numpy().T - Kstar))
        A.grad.zero_()
    with open('A_iterates_{}.npy'.format(dataset), 'wb') as f:
        np.save(f, np.array(A_iterates))
    print(iterate, dists[-1])
    # if dists[-1] < 0.01:
    #     break

pd.DataFrame(dists).to_csv(f"dists_{dataset}.csv")
with open(f"A_iterates_{dataset}.npy", "wb") as f:
    np.save(f, np.array(A_iterates))


