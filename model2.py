# model.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
class StockPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(60, 128)  # Input: 6 features * 10 minutes
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)    # Output: predicted close price
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def calculate_gain(y_test, outputs, X_test):
    # gain is tomorrow's close - today's close = y_test - output[-3]
    # for days where there was a predicted gain
    predicted_gain = (outputs - X_test[-3])
    # convert tensor to float
    predicted_gain = round(predicted_gain.item(), 4)
    if predicted_gain > 0:
        actual_gain = (y_test - X_test[-3])
        actual_gain = round(actual_gain.item(), 4)
        #print(f'Gain:real, predicted  -> {actual_gain}, {predicted_gain}')
        
        return actual_gain
    else:
        return 0

def calculate_gain_history(y_test, outputs, X_test, gain_history):
    #print(f'y_test shape: {y_test.shape}')
    gain_sum = 0
    for i in range(len(y_test)):
        gain_sum += calculate_gain(y_test[i], outputs[i], X_test[i])
    #print(f'Total gain for this test run: {gain_sum}')
    return gain_sum

# Train the model
def train_model(X_train, y_train, X_test, y_test):
    model = StockPredictor()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    train_loss_history = []
    test_loss_history = []
    test_gain_history = []
    # Train the model. Print the loss every epoch
    for epoch in range(10000):
        optimizer.zero_grad()
        outputs = model(X_train)
        train_loss = criterion(outputs, y_train)
        train_loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {train_loss.item()}")
            train_loss_history.append(train_loss.item())
            # test the model
            model.eval()
            with torch.no_grad():
                outputs = model(X_test)
                test_loss = criterion(outputs, y_test)
                #print(f"Test Loss: {test_loss.item()}")
                test_loss_history.append(test_loss.item())
                test_gain_history.append(calculate_gain_history(y_test, outputs, X_test, test_gain_history))
            model.train()
    # return the model and the loss history
    return model, train_loss_history, test_loss_history, test_gain_history

def read_data():
    X_train = np.loadtxt('X_train.csv', delimiter=',')
    y_train = np.loadtxt('y_train.csv', delimiter=',')
    X_test = np.loadtxt('X_test.csv', delimiter=',')
    y_test = np.loadtxt('y_test.csv', delimiter=',')
    # convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = read_data()
model, train_loss_history, test_loss_history, test_gain_history = train_model(X_train, y_train, X_train, y_train)
# plot the loss
plt.plot(train_loss_history)
plt.title('Training Loss over time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# plot the test loss
plt.plot(test_loss_history)
plt.title('Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
# plot the test gain history
plt.plot(test_gain_history)
plt.title('Real Gain, Test Data')
plt.xlabel('Tens of Epochs')
plt.ylabel('Gain')
plt.show()

# Test the model on the test data
def test_model(model, X_test, y_test):
    gain_history = []
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        outputs = model(X_test)
        loss = criterion(outputs, y_test)
        gain_history.append(calculate_gain_history(y_test, outputs, X_test, gain_history))
    return loss, gain_history

# Test the model
#test_loss, test_gain_history = test_model(model, X_test, y_test)


