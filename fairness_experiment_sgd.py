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
	data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
	acs_data = data_source.get_data(states=["TX"], download=True)

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
	    # if iterate % 1000 == -1 % 1000:
	    #     print(iterate + 1, loss, np.linalg.norm(A.detach().cpu().numpy() @ A.detach().cpu().numpy().T - Kstar))


	Ahat = A_iterates[-1]

	print("AA^T to Kstar:", np.linalg.norm(Ahat @ Ahat.T - Kstar))

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

	X_train_t = torch.Tensor(np.array(X_train))
	Y_train_t = torch.Tensor(np.array(Y_train))

	X_test_t = torch.Tensor(np.array(X_test))
	Y_test_t = torch.Tensor(np.array(Y_test))

	train_dataset = TrainDataset(X_train_t, Y_train_t)
	test_dataset = TrainDataset(X_test_t, Y_test_t)

	train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=8)
	test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=8)

	network_standard = NeuralNet()
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

	print("Standard Training Loss:", standard_loss)

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


	print("Fair Training Loss:", result.loss)

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

	return audit, audit_true, worst_ratio, worst_ratio_true

for n in range(100, 401, 25):
	print(n)
	print(fairness_experiment(n))

