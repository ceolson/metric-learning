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
    
    return X, M, S, y, Astar, Kstar

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

def L(A, y, M):
    yMts = torch.reshape(y, (-1, 1, 1)) * M
    yMtAs = torch.einsum('bij,jk->bik', yMts, A)
    yMtAATs = torch.einsum('bij,jk->bik', yMtAs, torch.transpose(A, 0, 1))
    TryMtAATs = torch.einsum('bii->b', yMtAATs)
    losses = torch.log(1 + torch.exp(-TryMtAATs))
    
    return torch.mean(losses)

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
print(np.shape(np.array(acs_data_cleaned)))


print("Generating synthetic data...")
X, M, S, y, Astar, Kstar = generate_synthetic_data(n, r, p, synthetic_data)
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
