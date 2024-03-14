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


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim, output_dim)
       
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        # x = torch.nn.functional.sigmoid(self.layer_2(x))
        x = self.layer_2(x)
        return x

def MSE_calculation(observed, pred):
    MSE = np.zeros(len(observed))
    n = len(observed)
    for i in range(n):
        MSE[i] = np.mean((observed - pred)**2)
    return MSE
    
def get_data(train_path, test_path):
    # X_train, X_test: first 30 columns, y_train: last col of train file, y_test: last column of test files
    train_set = pd.read_csv(train_path)
    test_set = pd.read_csv(test_path)
    
    X_train = train_set.iloc[:, :-1].values
    y_train = train_set.iloc[:, -1].values
    X_test = test_set.iloc[:, :-1].values
    y_test = test_set.iloc[:, -1].values

    X_train_tensor = torch.tensor(X_train).float()
    X_test_tensor = torch.tensor(X_test).float()
    y_train_tensor = torch.tensor(y_train).float()
    y_test_tensor = torch.tensor(y_test).float()
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)
    return train_dataset, train_dataloader,test_dataset, test_dataloader, X_test_tensor, y_train

    
    
def predict(num_epochs, input_dim, hidden_dim, output_dim):
    model = FeedForwardNeuralNetwork(input_dim, hidden_dim, output_dim)
    loss_fn = nn.CrossEntropyLoss()
    #stochastic gradient descent
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    iter = 0
    y_pred = []
    
    for epoch in range(num_epochs):
        for X, y in train_dataloader:
            # print(X.shape, y.shape)

            y = y.reshape(-1, 1)
            # zero the parameter gradients
            optimizer.zero_grad()
        
            # forward + backward + optimize
            outputs = model(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
    

    y_pred = []
    i = 0
    for X, y in test_dataloader:
        #print(X, y)
        # Forward pass only to get logits/output
        outputs = model(X)
        outputs = torch.squeeze(outputs)
        y_pred.append(outputs)
        # print(outputs)
        #print("pred2", pred)
        # Get predictions from the maximum value
        #predicted,_ = torch.max(pred.data, 1)
        #print("predicted", predicted)
    y_pred = torch.cat(y_pred, dim=0)
    print(y_pred)
    # print(y_train.shape)
    # #mse = MSE_calculation(y_train, y_pred)   
    # #print(mse)
    # #print("y_pred:", y_pred)
    return y_pred
    
    #use the model for classification
    
    # y_pred = []
    # model.eval()
    # with torch.no_grad():
    #     logits = model(X_test_tensor)  
    #     y_pred = torch.argmax(logits, dim=1).numpy()

    # print(y_pred)
    # return y_pred
    
    # y_pred = []
    # y_test = []
    # correct = total = 0
    # with torch.no_grad():
    #     for X, y in test_dataloader:
    #         outputs = model(X)
    #         predicted = np.where(outputs < 0.5, 0, 1)
    #         predicted = list(itertools.chain(*predicted))
    #         y_pred.append(predicted)
    #         y_test.append(y)
    #         total += y.size(0)
    #         correct += (predicted == y.numpy()).sum().item()

    # print(f'Accuracy of the network on the 3300 test instances: {100 * correct // total}%')

    
batch_size = 64

train_path = "data/original_processed/googl_train.csv"
test_path = "data/original_processed/googl_test_2022.csv"

train_dataset, train_dataloader, test_dataset, test_dataloader, X_test_tensor, y_train = get_data(train_path, test_path)
y_pred = predict(num_epochs=1000, input_dim = 30, hidden_dim = 10, output_dim = 1)