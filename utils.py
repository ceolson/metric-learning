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

class TrainDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return data, label

    def __len__(self):
        return len(self.labels)

class NeuralNet(torch.nn.Module):
    def __init__(self, p):
        super(NeuralNet, self).__init__()
        self.lin1 = torch.nn.Linear(p, 20, bias=False)
        self.lin2 = torch.nn.Linear(20, 20, bias=False)
        self.lin3 = torch.nn.Linear(20, 1, bias=False)

    def forward(self, x):
        x = torch.nn.functional.relu(self.lin1(x))
        x = torch.nn.functional.relu(self.lin2(x))
        return torch.nn.functional.sigmoid(self.lin3(x))

def relu(z):
    return z * (z > 0)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def softmax(z):
    exponentials = np.exp(z - np.max(z))
    return exponentials / np.sum(exponentials)

def m_dist2(x1, x2, K):
    return (x2 - x1).T @ K @ (x2 - x1)

def clean_data(n, p, data, cut_columns=True):
    X = data.head(n)
    X = X.loc[:, X.var(axis=0) > 0]
    if cut_columns:
        X = X.sample(p, axis=1)
    X = np.array(stats.zscore(np.array(X), axis=0))
    return X

def generate_synthetic_data(n, r, p, X, S=None):
    Astar = np.random.normal(0, 1, size=(p, r)) / (np.sqrt(p) * np.sqrt(r))
    Kstar = Astar @ Astar.T

    if not S:
        Sbar = []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i != j and j != k and k != i:
                        Sbar.append((i, j, k))
        Sbar = np.array(Sbar)
    
        choices = np.random.choice([False, True], size=len(Sbar), replace=True, p=[0.5, 0.5])
        S = Sbar[choices, :]

    M = []
    for t in S:
        i, j, k = t
        Mt = np.outer(X[i], X[k]) + np.outer(X[k], X[i]) - np.outer(X[i], X[j]) - np.outer(X[j], X[i]) + np.outer(X[j], X[j]) - np.outer(X[k], X[k])
        M.append(Mt)
    M = np.array(M)

    probs = 1 / (1 + np.exp(-np.einsum('ijj->i', np.einsum('ijk,kl->ijl', M, Kstar))))

    y = 2 * np.array(probs > np.random.random(np.shape(probs)), dtype=int) - 1
    
    # y = []
    # for t in S:
    #     i, j, k = t
    #     prob = 1 / (1 + np.exp(m_dist2(X[i], X[k], Kstar) - m_dist2(X[i], X[j], Kstar)))
    #     yt = np.random.choice([1, -1], size=1, p=[prob, 1 - prob])
    #     y.append(yt)
    # y = np.array(y)
    
    return M, S, y, Astar, Kstar

def initialization(n, r, p, S, X, y):
    transition_matrices = [np.zeros((n - 1, n - 1)) for _ in range(n)]
    for t in range(len(S)):
        def gap(l, i):
            if l < i:
                return l
            else:
                return l - 1
        i, j, k = S[t]
        transition_matrices[i][gap(j, i), gap(k, i)] = 0.99 if y[t] == -1 else 0.01
        transition_matrices[i][gap(k, i), gap(j, i)] = 0.99 if y[t] == 1 else 0.01

    for i in range(n):
        d = np.max(np.sum(transition_matrices[i], axis=1))
        transition_matrices[i] = transition_matrices[i] / d

        self_loops = np.diag(1 - np.sum(transition_matrices[i], axis=1))

        transition_matrices[i] += self_loops

    dists_from_i = []
    for i in range(n):
        eigenvalues, eigenvectors = eig(transition_matrices[i], left=True, right=False)
        leading_index = np.where(np.isclose(eigenvalues, 1))[0][0]
        leading_eigenvector = eigenvectors[:, leading_index].real
        # print(leading_eigenvector)
        dists_wo_i = np.log(np.maximum(leading_eigenvector, 0.001))
        if not np.all(np.isfinite(dists_wo_i)):
            print(i)
        dists = np.zeros(n)
        np.put(dists, list(range(i)) + list(range(i+1, n)), dists_wo_i)
        dists_from_i.append(dists)
    D = np.stack(dists_from_i)

    J = np.identity(n) - (np.ones((n, 1)) @ np.ones((1, n))) / n
    H = - J @ D @ J / 2
    Xprime = J @ X

    XtHX = Xprime.T @ H @ Xprime / (n**2)
    Sigma = Xprime.T @ Xprime / (n)
    eigenvalues, eigenvectors = eigh(a=XtHX, b=Sigma)
    U = eigenvectors[:, np.argsort(eigenvalues)[-r:]]
    Lambda = np.sort(eigenvalues)[-r:]
    Ahat = U @ np.diag(np.sqrt(Lambda))
    return Ahat

def L(A, y, M):
    yMts = torch.reshape(y, (-1, 1, 1)) * M
    yMtAs = torch.einsum('bij,jk->bik', yMts, A)
    yMtAATs = torch.einsum('bij,jk->bik', yMtAs, torch.transpose(A, 0, 1))
    TryMtAATs = torch.einsum('bii->b', yMtAATs)
    losses = torch.log(1 + torch.exp(-TryMtAATs))
    
    return torch.mean(losses)

def learn_fair_classifiers(X_train, Y_train, X_test, Y_test, Ahat, Astar):
    p = np.shape(np.array(X_train))[-1]
    X_train_t = torch.Tensor(np.array(X_train))
    Y_train_t = torch.Tensor(np.array(Y_train))

    X_test_t = torch.Tensor(np.array(X_test))
    Y_test_t = torch.Tensor(np.array(Y_test))

    train_dataset = TrainDataset(X_train_t, Y_train_t)
    test_dataset = TrainDataset(X_test_t, Y_test_t)

    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=8)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=8)

    network_standard = NeuralNet(p)
    optimizer = torch.optim.Adam(network_standard.parameters(), lr=1e-3)
    loss_fn = torch.nn.functional.binary_cross_entropy

    network_standard.train()

    for epoch in range(200):

        for x, y in train_dl:
            optimizer.zero_grad()
            y_pred = network_standard(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

    standard_loss = loss

    input_metric = MahalanobisDistances()
    input_metric.fit(torch.Tensor(Ahat @ Ahat.T))

    input_metric_true = MahalanobisDistances()
    input_metric_true.fit(torch.Tensor(Astar @ Astar.T))


    output_metric = SquaredEuclideanDistance()
    output_metric.fit(num_dims=1)


    network = NeuralNet(p)

    rho = 5.0
    eps = 0.1
    auditor_nsteps = 100
    auditor_lr = 0.001

    alg = SenSeI(network, input_metric, output_metric, loss_fn, rho, eps, auditor_nsteps, auditor_lr)

    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    alg.train()

    for epoch in range(4000):
        for x, y in train_dl:
            optimizer.zero_grad()
            result = alg(x, torch.reshape(y, (-1, 1)))
            result.loss.backward()
            optimizer.step()
        if result.loss < standard_loss:
            print("Stopping")
            break

    fair_loss = result.loss

    auditor = SenSeIAuditor(input_metric, output_metric, auditor_nsteps, auditor_lr)
    auditor_true = SenSeIAuditor(input_metric_true, output_metric, auditor_nsteps, auditor_lr)

    audit = auditor.audit(network, X_test_t, Y_test_t, torch.nn.functional.l1_loss)
    audit_true = auditor_true.audit(network, X_test_t, Y_test_t, torch.nn.functional.l1_loss)

    ratios = []
    for X_1 in X_test_t:
        for X_2 in X_test_t:
            ratios.append((output_metric(network(X_1), network(X_2)) / input_metric(X_1, X_2)).detach().numpy())
    ratios = np.array(ratios)
    worst_ratio = np.max(ratios[~np.isnan(ratios)])

    ratios = []
    for X_1 in X_test_t:
        for X_2 in X_test_t:
            ratios.append((output_metric(network(X_1), network(X_2)) / input_metric_true(X_1, X_2)).detach().numpy())
    ratios = np.array(ratios)
    worst_ratio_true = np.max(ratios[~np.isnan(ratios)])


    return standard_loss.detach().numpy(), fair_loss.detach().numpy(), audit, audit_true, worst_ratio, worst_ratio_true
