# Stock Price Prediction Prototype

This project implements a neural network-based stock price prediction model using PyTorch. The model takes stock data from the past 10 days (from kaggle) as input and predicts tomorrow's closing price. It plots the 'gain' calculated as gain = tomorrow's real closing price - today's real closing price but only for days where tomorrow's predicted closing price is higher than today's closing price. ie, if you trade on all the days where the model predicts a gain, how much do you make?

A few problems:
1. All the data has been normalized so the 'gains' are not in real $. I will fix this in a later version of some follow-on project.
2. I used an 80/20 split for training vs testing data so a training vector (which uses 10 days of data), might be days 100 through 109, predicting the closing price on day 110. But I may have a testing vector that uses days 99 through 108 as input, predicting the closing price on day 109 (which was already seen in the afore mentioned training vector). I don't really think this is a problem, but I will do another project (maybe prototype1) where I pretrain on 80%, test on the first day not in the training set, retrain using this new data point, then test on the next day, etc, until all the data has been used.

## Features

- Data preprocessing and normalization
- Neural network model with multiple fully connected layers
- Training and testing functionality
- Support for time series data with sliding windows

## Requirements

- Python 3.x
- PyTorch
- pandas
- numpy
- scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/pransil/prototype1.git
cd prototype1
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your data in the required format
2. Run the model:
```bash
python model.py
```

## Project Structure

- `model.py`: Contains the neural network model and training logic
- `prepare_data.py`: Data preprocessing utilities
- `requirements.txt`: Project dependencies

## License

MIT License 