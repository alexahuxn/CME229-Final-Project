import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.graphics.tsaplots import plot_acf
import os
import warnings
warnings.filterwarnings('ignore')

def load_data(ticker, start_date, end_date):
    data = yf.download(tickers=[ticker], start=start_date, end=end_date, interval="1d")
    data['Pct Change'] = data['Close'].pct_change() * 100
    data = data.dropna()
    data.drop(columns=['High', 'Low', 'Adj Close'], inplace =True)
    data.reset_index(inplace=True)
    return data

def save_plot_data(ticker, data):
    plt.figure(figsize=(9,6))
    plt.plot(data['Date'], data['Pct Change'])
    plt.title(f"Daily {ticker} Percentage Change", fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Percentage Change', fontsize=14)
    plt.grid(True)
    file_path = os.path.join('data_analysis', 'real_return.png')
    plt.savefig(file_path)
    plt.show()
    

def plot_autocorr(data, lag):
    fig, ax = plt.subplots(figsize=(9, 6))
    plot_acf(data, lags=lag, zero=False, ax=ax)
    return 


googl = load_data('GOOGL', '2006-01-01', '2023-12-31')
file_path = os.path.join('real_data', 'googl.csv')
googl.to_csv(file_path, index=False)
save_plot_data('Google', googl)