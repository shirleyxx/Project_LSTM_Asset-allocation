
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from sklearn.model_selection import train_test_split
from configs import *

n_epochs = 30
time_steps = rebalance_days
hidden_dim = 20
batch_size = 30
activation = None
loss = 'logcosh'

def create_input(data, time_steps=time_steps, data_dim=n_assets):     
    seqs = []
    for i in range(len(data)-time_steps):
        seqs.append(data[i: i+time_steps]) 
    return np.asarray(seqs)

class LSTM_model():
    def __init__(self):
        self.model = Sequential()
    
    def build(self, 
              time_steps = time_steps-1, 
              data_dim   = n_assets, 
              output_dim = n_assets, 
              hidden_dim = hidden_dim):
        # expected input batch shape: (batch_size, timesteps, data_dim)
        # the sample of index i in batch k is the follow-up for the sample i in batch k-1.
        self.model.add(LSTM(hidden_dim, return_sequences=True, input_shape=(time_steps, data_dim)))
        self.model.add(LSTM(hidden_dim, return_sequences=True))
        self.model.add(LSTM(hidden_dim))
        self.model.add(Dense(output_dim, activation = activation)) 
        self.model.compile(loss = loss, optimizer='rmsprop', metrics=['accuracy']) 
        return self.model

    def train(self, seqs, split=0.1, plot=False):
        x = seqs[:, : -1]
        y = seqs[:, -1]
        
        size = len(seqs)
        x = x[: batch_size * (size // batch_size)]
        y = y[: batch_size * (size // batch_size)]
        
        if len(x.shape) == 2:
            x = x.reshape(x.shape[0], x.shape[1], 1)
        if len(y.shape) == 1:
            y = y.reshape(y.shape[0],1)

        print(x.shape, y.shape)
        self.model.fit(x, y, batch_size=batch_size, epochs=n_epochs, validation_split=0.1, verbose=1)
        if plot:
            self.train_plot = self.view_accuracy(self.predict(x), y, 'Train')
            
    def predict(self, x_test):
        return self.model.predict(x_test)
    
    def view_accuracy(self, y_pred=None, y_true=None, plot_name='Test'):
        if y_pred is None:
            y_pred = self.y_pred
            y_true = self.y_test_true
            
        plt.style.use('seaborn')
        plt.figure(figsize=(10,6))
        plt.grid(True)
        plt.plot(y_pred, color='lightcoral')
        plt.plot(y_true, color='cornflowerblue', linewidth=1)
        plt.title(plot_name)
        plt.legend(['predict', 'true'])

