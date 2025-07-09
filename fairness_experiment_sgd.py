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
        "AA^T to Kstar", 
        "Standard Train Loss", 
        "Fair Train Loss", 
        "Audit Mean", 
        "Audit Std", 
        "Audit Lower Bound",
        "True Audit Mean", 
        "True Audit Std", 
        "True Audit Lower Bound",
        "Worst Ratio", 
        "Worst Ratio True"
    ])

features, labels, _ = ACSMobility.df_to_pandas(acs_data)

X = clean_data(2 * n, len(features.columns), features, cut_columns=False)
X_train = X[:n]
X_test = X[n:]

Y = labels.head(2 * n)
Y_train = Y[:n]
Y_test = Y[n:]

p = np.shape(X)[-1]

Astar = np.random.normal(0, 1, size=(p, r)) / (np.sqrt(p) * np.sqrt(r))
Kstar = Astar @ Astar.T

A0 = np.random.normal(0, 1, size=(p, r)) / (np.sqrt(p) * np.sqrt(r))

A_iterates = []
A = torch.tensor(A0, requires_grad=True, device="cpu").to(torch.float64)
dists = []

for iterate in range(20000):
    S = np.random.choice(range(X_train.shape[0]), replace=True, size=(1000, 3))
    M = []
    for t in S:
        i, j, k = t
        Mt = np.outer(X_train[i], X[k]) + np.outer(X_train[k], X[i]) \
            - np.outer(X_train[i], X[j]) - np.outer(X_train[j], X[i]) \
            + np.outer(X_train[j], X[j]) - np.outer(X_train[k], X[k])
        M.append(Mt)
    M = np.array(M)
    
    probs = 1 / (1 + np.exp(-np.einsum('ijj->i', np.einsum('ijk,kl->ijl', M, Kstar))))

    y = 2 * np.array(probs > np.random.random(np.shape(probs)), dtype=int) - 1
    
    loss = L(A, torch.Tensor(y), torch.Tensor(M).to(torch.float64))
    loss.backward()
    with torch.no_grad():
        A -= A.grad * 0.1
        A_iterates.append(A.detach().cpu().numpy())
        dists.append(np.linalg.norm(A.detach().cpu().numpy() @ A.detach().cpu().numpy().T - Kstar))
        A.grad.zero_()
	    
	if iterate % 50 == -1 % 50:
        print(iterate + 1, loss, dists[-1])
        standard_loss, fair_loss, audit, audit_true, worst_ratio, worst_ratio_true = learn_fair_classifiers(
        	X_train, Y_train, X_test, Y_test, A.detach().cpu().numpy(), Astar)
        
        results.loc[len(results.index)] = {
            "Iterate": iterate,
            "AA^T to Kstar": dists[-1], 
            "Standard Train Loss": standard_loss, 
            "Fair Train Loss": fair_loss, 
            "Audit Mean": audit.lossratio_mean, 
            "Audit Std": audit.lossratio_std, 
            "Audit Lower Bound": audit.lower_bound,
            "True Audit Mean": audit_true.lossratio_mean, 
            "True Audit Std": audit_true.lossratio_std, 
            "True Audit Lower Bound": audit_true.lower_bound,
            "Worst Ratio": worst_ratio, 
            "Worst Ratio True": worst_ratio_true
        }

        results.to_csv("fairness_during_gd_out.csv")

Ahat = A_iterates[-1]

print("AA^T to Kstar:", np.linalg.norm(Ahat @ Ahat.T - Kstar))

results.to_csv("fairness_during_gd_out.csv")

