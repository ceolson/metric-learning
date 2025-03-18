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
    def __init__(self):
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


r = 3
p = 30

def fairness_experiment(n):
    synthetic_data = pd.DataFrame(np.random.normal(0, 1, size=(2 * n, p)))

    X = clean_data(2 * n, p, synthetic_data)
    X_train = X[:n]
    X_test = X[n:]

    print("Generating synthetic data...")
    M, S, y, Astar, Kstar = generate_synthetic_data(n, r, p, X_train)

    print("Initialization...")
    A0 = initialization(n, r, p, S, X_train, y)

    print("Gradient descent...")
    A_iterates = []
    A = torch.tensor(A0, requires_grad=True, device="cpu")
    dists = []

    y_tensor = torch.tensor(y, device="cpu")
    M_tensor = torch.tensor(M, device="cpu")
    for iterate in range(10 * n):
        loss = L(A, y_tensor, M_tensor)
        loss.backward()
        with torch.no_grad():
            A -= A.grad * 0.2
            A_iterates.append(A.detach().cpu().numpy())
            dists.append(np.linalg.norm(A.detach().cpu().numpy() @ A.detach().cpu().numpy().T - Kstar))
            A.grad.zero_()
        if iterate % 10 == 9:
            print(iterate + 1, loss, np.linalg.norm(A.detach().cpu().numpy() @ A.detach().cpu().numpy().T - Kstar))

    Ahat = A_iterates[-1]

    l2_error_A = np.linalg.norm(Ahat @ Ahat.T - Kstar)

    print("||Ahat @ Ahat.T - Kstar||:", l2_error_A)

    print("Generating synthetic targets for learning task...")
    Y_probs = np.sign(X[:, 0]) / 10 + 0.5

    Y = (np.random.random(size=2 * n) < Y_probs).astype(int)

    Y_train = Y[:n]
    Y_test = Y[n:]

    X_train_t = torch.Tensor(X_train)
    Y_train_t = torch.Tensor(Y_train)
    
    X_test_t = torch.Tensor(X_test)
    Y_test_t = torch.Tensor(Y_test)
    
    train_dataset = TrainDataset(X_train_t, Y_train_t)
    test_dataset = TrainDataset(X_test_t, Y_test_t)
    
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=8)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=8)

    print("Standard training algorithm...")
    network_standard = NeuralNet()
    optimizer = torch.optim.Adam(network_standard.parameters(), lr=1e-3)
    loss_fn = torch.nn.functional.binary_cross_entropy

    network_standard.train()

    for epoch in range(1000):

        for x, y in train_dl:
            optimizer.zero_grad()
            y_pred = network_standard(x).squeeze()
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

    print("Standard training loss:", loss)

    input_metric = MahalanobisDistances()
    input_metric.fit(torch.Tensor(Ahat @ Ahat.T))

    input_metric_true = MahalanobisDistances()
    input_metric_true.fit(torch.Tensor(Astar @ Astar.T))

    output_metric = SquaredEuclideanDistance()
    output_metric.fit(num_dims=1)

    network = NeuralNet()

    rho = 5.0
    eps = 0.1
    auditor_nsteps = 100
    auditor_lr = 0.001

    alg = SenSeI(network, input_metric, output_metric, loss_fn, rho, eps, auditor_nsteps, auditor_lr)

    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    print("Fair training algorithm...")

    alg.train()

    for epoch in range(2000):
        for x, y in train_dl:
            optimizer.zero_grad()
            result = alg(x, torch.reshape(y, (-1, 1)))
            result.loss.backward()
            optimizer.step()

    print("Fair training loss:", result.loss)

    print("Auditing...")

    auditor = SenSeIAuditor(input_metric, output_metric, auditor_nsteps, auditor_lr)
    auditor_true = SenSeIAuditor(input_metric_true, output_metric, auditor_nsteps, auditor_lr)

    audit = auditor.audit(network, X_test_t, torch.reshape(Y_test_t, (-1, 1)), torch.nn.functional.l1_loss)

    true_audit = auditor_true.audit(network, X_test_t, torch.reshape(Y_test_t, (-1, 1)), torch.nn.functional.l1_loss)

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
    true_worst_ratio = np.max(ratios[~np.isnan(ratios)])

    return l2_error_A, audit, true_audit, worst_ratio, true_worst_ratio

for n in [200, 250, 300, 350, 400, 450, 500]:
    print(n)
    print(fairness_experiment(n))
