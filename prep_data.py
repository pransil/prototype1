# prep_data.py
import pandas as pd
import numpy as np
import torch
import random
from sklearn.preprocessing import MinMaxScaler

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
    # round valuse in np arrays to 4 decimal places
    X_train = np.round(np.array(X_train), 4)
    y_train = np.round(np.array(y_train), 4)
    X_test = np.round(np.array(X_test), 4)
    y_test = np.round(np.array(y_test), 4)
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
   
    print(f"X_train[0]: {X_train[0]}")
    print(f"y_train[0]: {y_train[0]}")
    print(f"X_test[0]: {X_test[0]}")
    print(f"y_test[0]: {y_test[0]}")
    
    return X_train, y_train, X_test, y_test

# read in file tesla.csv
data = prepare_data('../stock_data/tesla.csv')
X_train, y_train, X_test, y_test = convert_data(data)
# save to X_train.csv, y_train.csv, X_test.csv, y_test.csv
np.savetxt('X_train.csv', X_train, delimiter=',')
np.savetxt('y_train.csv', y_train, delimiter=',')
np.savetxt('X_test.csv', X_test, delimiter=',')
np.savetxt('y_test.csv', y_test, delimiter=',')

