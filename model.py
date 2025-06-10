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

 #Prepare data - read the data from the file, drop ticker and window_start columns, 
 # convert to numpy array, normalize the data, return the data, the scaler, and the data shape
def prepare_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop(columns=['Date'])
    data = data.to_numpy()
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data

# Convert the ouput from prepare_data() to training and test sets. 
# Each input is ten lines of data where the target is the close for the 11th line.
# Return the training and test sets, the scaler, and the data shape
def convert_data(data):
    X = []
    y = []
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for i in range(len(data) - 11):
        X.append(data[i:i+10].flatten())  # Flatten the 10x6 window into a 60-dimensional vector
        # y is the close for the 11th line (tomorrow's close)
        y.append(data[i+11, 3]) 
        # random, approximately 80 / 20 split
        if random.random() < 0.8:
            X_train.append(X[i])
            y_train.append(y[i])
        else:
            X_test.append(X[i])
            y_test.append(y[i])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
   
    print(f"X_train[0]: {X_train[0]}")
    print(f"y_train[0]: {y_train[0]}")
    print(f"X_test[0]: {X_test[0]}")
    print(f"y_test[0]: {y_test[0]}")
    # Convert X, y to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    return X_train, y_train, X_test, y_test

def calculate_gain(y_test, outputs, X_test):
    # gain is tomorrow's close - today's close = y_test - output[-3]
    # for days where there was a predicted gain
    predicted_gain = (outputs - X_test[-3])
    # convert tensor to float
    predicted_gain = round(predicted_gain.item(), 4)
    if predicted_gain > 0:
        actual_gain = (y_test - X_test[-3])
        actual_gain = round(actual_gain.item(), 4)
        print(f'Gain:real, predicted  -> {actual_gain}, {predicted_gain}')
        
        return actual_gain
    else:
        return 0

def calculate_gain_history(y_test, outputs, X_test, gain_history):
    print(f'y_test shape: {y_test.shape}')
    gain_sum = 0
    for i in range(len(y_test)):
        gain_sum += calculate_gain(y_test[i], outputs[i], X_test[i])
    print(f'Total gain for this test run: {gain_sum}')
    return gain_sum

# Train the model
def train_model(X_train, y_train, X_test, y_test):
    model = StockPredictor()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    loss_history = []
    gain_history = []
    # Train the model. Print the loss every epoch
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            loss_history.append(loss.item())
            # test the model
            model.eval()
            with torch.no_grad():
                outputs = model(X_test)
                loss = criterion(outputs, y_test)
                print(f"Test Loss: {loss.item()}")
                gain_history.append(calculate_gain_history(y_test, outputs, X_test, gain_history))
            model.train()
    # return the model and the loss history
    return model, loss_history, gain_history


# After saving data to 'stock_data.csv'
data = prepare_data('../stock_data/tesla.csv')
X_train, y_train, X_test, y_test = convert_data(data)
model, loss_history, gain_history = train_model(X_train, y_train, X_train, y_train)
# plot the loss
plt.plot(loss_history)
plt.title('Loss over time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
# plot the gain history
plt.plot(gain_history)
plt.title('Gain over time')
plt.xlabel('Epoch')
plt.ylabel('Gain')
plt.show()

# Test the model
def test_model(model, X_test, y_test):
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        outputs = model(X_test)
        loss = criterion(outputs, y_test)
        return loss
