import numpy as np
from folktables import ACSDataSource, ACSMobility
from matplotlib import pyplot as plt
import torch
from scipy import stats
from scipy.sparse.linalg import lobpcg
import pandas as pd

def m_dist2(x1, x2, K):
    return (x2 - x1).T @ K @ (x2 - x1)

def generate_synthetic_data(n, r, p, data, S=None):
    X = data.sample(n, axis=0)
    X = X.loc[:, X.var(axis=0) > 0]
    X = X.sample(p, axis=1)
    X = np.array(stats.zscore(np.array(X), axis=0))
   
    Astar = np.random.normal(0, 1, size=(p, r)) / np.sqrt(p)
    Kstar = Astar @ Astar.T

    if not S:
        Sbar = []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i != j and j != k and k != i:
                        Sbar.append((i, j, k))
        Sbar = np.array(Sbar)
    
        choices = np.random.choice([False, True], size=len(Sbar), replace=True, p=[0.95, 0.05])
        S = Sbar[choices, :]

    M = []
    for t in S:
        i, j, k = t
        Mt = np.outer(X[i], X[k]) + np.outer(X[k], X[i]) - np.outer(X[i], X[j]) - np.outer(X[j], X[i]) + np.outer(X[j], X[j]) - np.outer(X[k], X[k])
        M.append(Mt)
    M = np.array(M)

    probs = 1 / (1 + np.exp(-np.einsum('ijj->i', np.einsum('ijk,kl->ijl', M,  Kstar))))

    y = 2 * np.array(probs > np.random.random(np.shape(probs)), dtype=int) - 1
    
    # y = []
    # for t in S:
    #     i, j, k = t
    #     prob = 1 / (1 + np.exp(m_dist2(X[i], X[k], Kstar) - m_dist2(X[i], X[j], Kstar)))
    #     yt = np.random.choice([1, -1], size=1, p=[prob, 1 - prob])
    #     y.append(yt)
    # y = np.array(y)
    
    return X, S, y, Astar, Kstar

def initialization(n, p, S, X, y):
    transition_matrices = [np.zeros((n - 1, n - 1)) for _ in range(n)]
    for t in range(len(S)):
        def gap(l, i):
            if l < i:
                return l
            else:
                return l - 1
        i, j, k = S[t]
        transition_matrices[i][gap(j, i), gap(k, i)] = 0.99 if y[t] == 1 else 0.01
        transition_matrices[i][gap(k, i), gap(j, i)] = 0.99 if y[t] == -1 else 0.01

    for i in range(n):
        d = np.max(np.sum(transition_matrices[i], axis=1))
        transition_matrices[i] = transition_matrices[i] / d

        self_loops = np.diag(1 - np.sum(transition_matrices[i], axis=1))

        transition_matrices[i] += self_loops

    dists_from_i = []
    for i in range(n):
        eigenvalues, eigenvectors = np.linalg.eigh(transition_matrices[i])
        leading_index = np.where(np.isclose(eigenvalues, 1))[0]
        leading_eigenvector = np.abs(eigenvectors[:, leading_index].real)
        dists_wo_i = np.log(leading_eigenvector)
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
    Sigma = Xprime.T @ Xprime / n
    eigenvalues, eigenvectors = lobpcg(A=XtHX, B=Sigma, X=np.random.normal(size=(p,r)))
    U = eigenvectors[:, np.argsort(eigenvalues)[-r:]]
    Lambda = np.sort(eigenvalues)[-r:]
    AAt = U @ np.diag(Lambda) @ U.T
    
    V, s, W = np.linalg.svd(AAt)
    
    top_r_indicies = np.argsort(s)[:r]
    V_r = V[:, top_r_indicies].reshape((p, r))
    s_r = s[top_r_indicies]
    Ahat = V_r @ np.sqrt(np.diag(s_r))

    return Ahat


synthetic_data = pd.DataFrame(np.random.normal(size=(1000, 50)))

r = 3
p = 30

for n in [250, 500, 750, 1000, 1250]:
    
    print("Generating synthetic data...")
    X, S, y, Astar, Kstar = generate_synthetic_data(n, r, p, synthetic_data)
    print(np.shape(X))
    
    np.save("Astar.npy", Astar)
    
    print("Initializing...")
    A0 = initialization(n, p, S, X, y)
    
    print(n)
    print(np.linalg.norm(Astar - A0))
