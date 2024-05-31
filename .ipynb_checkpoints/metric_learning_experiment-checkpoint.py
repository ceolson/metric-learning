import numpy as np
from scipy import stats, linalg
from folktables import ACSDataSource, ACSMobility
from matplotlib import pyplot as plt
import torch

def m_dist2(x1, x2, K):
    return (x2 - x1).T @ K @ (x2 - x1)

def generate_synthetic_data(n, r):
    np.random.shuffle(features)
    features_cut = features[:n]
    variances = np.var(features_cut, axis=0)
    features_nontrivial = np.reshape(features_cut[:, np.nonzero(variances)], (n, -1))
    _, p = np.shape(features_nontrivial)

    X = stats.zscore(features_nontrivial[:n], axis=0)
    
    Astar = np.random.normal(0, 1, size=(p, r)) / np.math.sqrt(p)
    Kstar = Astar @ Astar.T
    
    Sbar = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if i != j and j != k and k != i:
                    Sbar.append((i, j, k))
    Sbar = np.array(Sbar)
    
    choices = np.random.choice([False, True], p=[0.5, 0.5], size=len(Sbar))
    S = Sbar[choices, :]
    
    y = []
    for t in S:
        i, j, k = t
        prob = 1 / (1 + np.math.exp(m_dist2(X[i], X[k], Kstar) - m_dist2(X[i], X[j], Kstar)))
        yt = np.random.choice([1, -1], p=[prob, 1 - prob])
        y.append(yt)
    y = np.array(y)
    
    return p, X, S, y, Astar, Kstar

def initialization(n, p, S, y):
    transition_matrices = [np.zeros((n - 1, n - 1)) for _ in range(n)]
    for t in range(len(S)):
        def gap(l, i):
            if l < i:
                return l
            else:
                return l - 1
        i, j, k = S[t]
        transition_matrices[i][gap(j, i), gap(k, i)] = 0.9 if y[t] == 1 else 0.1
        transition_matrices[i][gap(k, i), gap(j, i)] = 0.9 if y[t] == -1 else 0.1

    for i in range(n):
        d = np.max(np.sum(transition_matrices[i], axis=1))
        transition_matrices[i] = transition_matrices[i] / d

        self_loops = np.diag(1 - np.sum(transition_matrices[i], axis=1))

        transition_matrices[i] += self_loops

    dists_from_i = []
    for i in range(n):
        eigenvalues, eigenvectors = linalg.eig(transition_matrices[i], left=True, right=False)
        leading_index = np.where(np.isclose(eigenvalues, 1))[0]
        leading_eigenvector = np.abs(eigenvectors[:, leading_index].real)
        dists_wo_i = np.log(leading_eigenvector)
        if not np.all(np.isfinite(dists_wo_i)):
            print(i)
        dists = np.insert(dists_wo_i, i, 0)
        dists_from_i.append(dists)
    D = np.stack(dists_from_i)
    Dtilde = D + D.T / 2
    
    J = np.identity(n) - (np.ones((n, 1)) @ np.ones((1, n))) / n
    H = - J @ Dtilde @ J / 2
    
    XtHX = X.T @ H @ X
    Sigma = X.T @ X
    eigenvalues, eigenvectors = linalg.eig(a=XtHX, b=Sigma)
    U = eigenvectors[:, np.argsort(eigenvalues)[-r:]]
    Lambda = np.sort(eigenvalues)[-r:]
    AAt = U @ np.diag(Lambda) @ U.T
    
    V, s, W = linalg.svd(AAt)
    
    top_r_indicies = np.argsort(s)[:r]
    V_r = V[:, top_r_indicies].reshape((p, r))
    s_r = s[top_r_indicies]
    Ahat = V_r @ np.sqrt(np.diag(s_r))

    return Ahat

def L(A):
    triplets = torch.tensor(signed_M, requires_grad=True)
    
    yMtAs = torch.einsum('bij,jk->bik', triplets, A)
    yMtAATs = torch.einsum('bij,jk->bik', yMtAs, torch.transpose(A, 0, 1))
    TryMtAATs = torch.einsum('bii->b', yMtAATs)
    losses = torch.log(1 + torch.exp(-TryMtAATs))
    
    return torch.mean(losses)


if __name__ == '__main__':

    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["CA"], download=True)
    features, label, group = ACSMobility.df_to_numpy(acs_data)

    r = 3
    n = 300
     
    print("Generating synthetic data...")   
    p, X, S, y, Astar, Kstar = generate_synthetic_data(n, r)

    print("Initializing...")
    A0 = initialization(n, p, S, y)

    M = []
    for t in S:
        i, j, k = t
        Mt = np.outer(X[i], X[k]) + np.outer(X[k], X[i]) - np.outer(X[i], X[j]) - np.outer(X[j], X[i]) + np.outer(X[j], X[j]) - np.outer(X[k], X[k])
        M.append(Mt)
    M = np.array(M)

    signed_M = np.multiply(M, np.reshape(y, (-1, 1, 1)))

    A_iterates = []
    A = torch.tensor(A0.real, requires_grad=True)
    dists = []

    print("Starting gradient descent...")
    for iterate in range(100):
        loss = L(A)
        loss.backward()
        with torch.no_grad():
            A -= A.grad * 0.1
            A_iterates.append(A.detach().numpy())
            dists.append(np.linalg.norm(A.detach().numpy() @ A.detach().numpy().T - Kstar))
            A.grad.zero_()
        if iterate % 10 == 9:
            print(iterate + 1, loss, np.linalg.norm(A.detach().numpy() @ A.detach().numpy().T - Kstar))

    plt.plot(dists)
    plt.xlabel("iteration")
    plt.ylabel("||AA^T - K*||")

    plt.savefig("gd.png")

