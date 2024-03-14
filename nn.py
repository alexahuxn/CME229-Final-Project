import torch
import pandas as pd
import numpy as np
from torch import nn
from torch import optim
import itertools
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim, output_dim)
       
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.sigmoid(self.layer_2(x))
        return x
       
    
def get_data(train_path, test_path):
    # X_train, X_test: first 30 columns, y_train: last col of train file, y_test: last column of test files
    train_set = pd.read_csv(train_path)
    test_set = pd.read_csv(test_path)
    
    print(train_set.shape)
    print(test_set.shape)
    
    X_train = train_set.iloc[:, :-1].values
    y_train = train_set.iloc[:, -1].values
    X_test = test_set.iloc[:, :-1].values
    y_test = test_set.iloc[:, -1].values
    
    print(X_train.shape)
    print(y_train.shape)
    
    X_train_tensor = torch.tensor(X_train).float()
    X_test_tensor = torch.tensor(X_test).float()
    y_train_tensor = torch.tensor(y_train).float()
    y_test_tensor = torch.tensor(y_test).float()
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)
    return train_dataset, train_dataloader,test_dataset, test_dataloader 

    
    
def train(num_epochs, input_dim, hidden_dim, output_dim):
    model = NeuralNetwork(input_dim, hidden_dim, output_dim)
    loss_fn = nn.BCELoss()
    #stochastic gradient descent
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    loss_values = []

    for epoch in range(num_epochs):
        for X, y in train_dataloader:
            # zero the parameter gradients
            optimizer.zero_grad()
        
            # forward + backward + optimize
            pred = model(X)
            loss = loss_fn(pred, y.unsqueeze(-1))
            loss_values.append(loss.item())
            loss.backward()
            optimizer.step()

    print("Training Complete")
    
    y_pred = []
    y_test = []
    correct = total = 0
    with torch.no_grad():
        for X, y in test_dataloader:
            outputs = model(X)
            predicted = np.where(outputs < 0.5, 0, 1)
            predicted = list(itertools.chain(*predicted))
            y_pred.append(predicted)
            y_test.append(y)
            total += y.size(0)
            correct += (predicted == y.numpy()).sum().item()

    print(f'Accuracy of the network on the 3300 test instances: {100 * correct // total}%')

    
batch_size = 64

train_path = "data/original_processed/googl_train.csv"
test_path = "data/original_processed/googl_test_2018.csv"

train_dataset, train_dataloader, test_dataset, test_dataloader = get_data(train_path, test_path)
train(num_epochs=100, input_dim = 2, hidden_dim = 10, output_dim = 1)
