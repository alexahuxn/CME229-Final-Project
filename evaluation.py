#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import torch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

def construct_price_from_pct(pct_array, initial_value):
    price = np.zeros(len(pct_array)+1)
    price[0] = initial_value
    for i in range(1, len(pct_array)+1):
        price[i] = price[i-1]*(1+pct_array[i-1]/100)
    return price

def MSE_calculation(data, start_idx, end_idx, col0, *cols):
    df = data.iloc[start_idx:end_idx]
    MSE = np.zeros(len(cols))
    for i, col in enumerate(cols):
        MSE[i] = np.mean((df[col0]-df[col])**2)
        
    plt.figure(figsize=(12,6))
    plt.bar(cols, MSE)
    plt.title("MSE by Models")
    plt.xlabel('Models', fontsize=14)
    plt.ylabel('Mean Square Error', fontsize=14)
    plt.grid(True)
    plt.show()
    return MSE

def month_MSE(data, start_month, end_month, col0, *cols):
    month_range = pd.date_range(start=start_month, end=end_month, freq='MS')
    MSE = np.zeros((len(month_range),len(cols)))
    start_year = int(start_month[:4]) 
    for j in range(len(cols)):
        for i in range(len(month_range)):
            df = data[(data['Date'].dt.year == start_year+i//12) & (data['Date'].dt.month == i%12+1)]
            MSE[i,j] = np.mean((df[col0]-df[cols[j]])**2)
    
    plt.figure(figsize=(12,6))
    for j in range(len(cols)):
        plt.plot(month_range, MSE[:,j], marker='o', label=cols[j])
    plt.title('MSE of Prediction', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Mean Square Error', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()
    return MSE

def train_ARIMA(x, order):
    model = ARIMA(x, order=order)
    model_fit = model.fit()
    return model_fit

def predict_ARIMA(data, window_size, col_val, col_chg, order, start, end, model_trained):
    pct_prediction = np.zeros(len(data))
    price_prediction = np.zeros(len(data))
    for i in range(start, end):
        df = data.iloc[i-window_size+1:i+1]
        x = df[col_chg]
        model = ARIMA(x, order=order)
        model_fit = model.fit()
        model_fit.arparams = model_trained.arparams
        print(model_fit.arparams)
        print(model_fit.summary())
        pct_prediction[i] = model_fit.forecast(steps=1)
        price_prediction[i] = (1+pct_prediction[i]/100) * data.iloc[i-1][col_val]
    return pct_prediction, price_prediction

