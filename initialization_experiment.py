import numpy as np
from folktables import ACSDataSource, ACSMobility
from matplotlib import pyplot as plt
import torch
import cupy as cp
from cupyx.scipy import stats
from cupyx.scipy.sparse.linalg import lobpcg
import pandas as pd

def m_dist2(x1, x2, K):
    return (x2 - x1).T @ K @ (x2 - x1)

def generate_synthetic_data(n, r, p, data, S=None):
    X = data.sample(n, axis=0)
    X = X.loc[:, X.var(axis=0) > 0]
    X = X.sample(p, axis=1)
    X = cp.array(stats.zscore(cp.array(X), axis=0))
   
    Astar = cp.random.normal(0, 1, size=(p, r)) / cp.sqrt(p)
    Kstar = Astar @ Astar.T

    if not S:
        Sbar = []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i != j and j != k and k != i:
                        Sbar.append((i, j, k))
        Sbar = cp.array(Sbar)
    
        choices = cp.random.choice([False, True], size=len(Sbar), replace=True, p=[0.7, 0.3])
        S = Sbar[choices, :]

    M = []
    for t in S:
        i, j, k = t
        Mt = cp.outer(X[i], X[k]) + cp.outer(X[k], X[i]) - cp.outer(X[i], X[j]) - cp.outer(X[j], X[i]) + cp.outer(X[j], X[j]) - cp.outer(X[k], X[k])
        M.append(Mt)
    M = cp.array(M)

    probs = 1 / (1 + cp.exp(-cp.einsum('ijj->i', cp.einsum('ijk,kl->ijl', M,  Kstar))))

    y = 2 * cp.array(probs > cp.random.random(cp.shape(probs)), dtype=int) - 1
    
    # y = []
    # for t in S:
    #     i, j, k = t
    #     prob = 1 / (1 + cp.exp(m_dist2(X[i], X[k], Kstar) - m_dist2(X[i], X[j], Kstar)))
    #     yt = cp.random.choice([1, -1], size=1, p=[prob, 1 - prob])
    #     y.append(yt)
    # y = cp.array(y)
    
    return X, S, y, Astar, Kstar

def initialization(n, p, S, X, y):
    transition_matrices = [cp.zeros((n - 1, n - 1)) for _ in range(n)]
    for t in range(len(S)):
        def gap(l, i):
            if l < i:
                return l
            else:
                return l - 1
        i, j, k = S[t]
        i = i.get()
        j = j.get()
        k = k.get()
        transition_matrices[i][gap(j, i), gap(k, i)] = 0.99 if y[t] == 1 else 0.01
        transition_matrices[i][gap(k, i), gap(j, i)] = 0.99 if y[t] == -1 else 0.01

    for i in range(n):
        d = cp.max(cp.sum(transition_matrices[i], axis=1))
        transition_matrices[i] = transition_matrices[i] / d

        self_loops = cp.diag(1 - cp.sum(transition_matrices[i], axis=1))

        transition_matrices[i] += self_loops

    dists_from_i = []
    for i in range(n):
        eigenvalues, eigenvectors = cp.linalg.eigh(transition_matrices[i])
        leading_index = cp.where(cp.isclose(eigenvalues, 1))[0]
        leading_eigenvector = cp.abs(eigenvectors[:, leading_index].real)
        dists_wo_i = cp.log(leading_eigenvector)
        if not cp.all(cp.isfinite(dists_wo_i)):
            print(i)
        dists = cp.zeros(n)
        cp.put(dists, list(range(i)) + list(range(i+1, n)), dists_wo_i)
        dists_from_i.append(dists)
    D = cp.stack(dists_from_i)

    J = cp.identity(n) - (cp.ones((n, 1)) @ cp.ones((1, n))) / n
    H = - J @ D @ J / 2
    Xprime = J @ X
    
    XtHX = Xprime.T @ H @ Xprime / (n**2)
    Sigma = Xprime.T @ Xprime / n
    eigenvalues, eigenvectors = lobpcg(A=XtHX, B=Sigma, X=cp.random.normal(size=(p,r)))
    U = eigenvectors[:, cp.argsort(eigenvalues)[-r:]]
    Lambda = cp.sort(eigenvalues)[-r:]
    AAt = U @ cp.diag(Lambda) @ U.T
    
    V, s, W = cp.linalg.svd(AAt)
    
    top_r_indicies = cp.argsort(s)[:r]
    V_r = V[:, top_r_indicies].reshape((p, r))
    s_r = s[top_r_indicies]
    Ahat = V_r @ cp.sqrt(cp.diag(s_r))

    return Ahat

def L(A, triplets): 
    yMtAs = torch.einsum('bij,jk->bik', triplets, A)
    yMtAATs = torch.einsum('bij,jk->bik', yMtAs, torch.transpose(A, 0, 1))
    TryMtAATs = torch.einsum('bii->b', yMtAATs)
    losses = torch.log(1 + torch.exp(-TryMtAATs))
    
    return torch.mean(losses)

synthetic_data = pd.DataFrame(cp.random.normal(size=(1000, 50)).get())

r = 3
p = 20

for n in range(100, 400, 20):
    
    print("Generating synthetic data...")
    X, S, y, Astar, Kstar = generate_synthetic_data(n, r, p, synthetic_data)
    print(cp.shape(X))
    
    cp.save("Astar.npy", Astar.get())
    
    print("Initializing...")
    A0 = initialization(n, p, S, X, y)
    
    print(n)
    print(cp.linalg.norm(Astar - A0))
