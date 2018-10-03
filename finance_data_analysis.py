# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 16:07:10 2018

@author: Wuethrich Pierre

This is an example of Data Analysis in the sector of Finance using Pandas, NumPy,
Seaborn and Data from the Google Finance/IEX website
"""
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt

#Getting the Data from Google Finance/IEX database using the Pandas-datareader

start = datetime.datetime(2015, 1, 1)
end = datetime.datetime(2018, 1, 1)

BAC = web.DataReader('BAC', 'iex', start, end)
C = web.DataReader('C', 'iex', start, end)
GS = web.DataReader('GS', 'iex', start, end)
JPM = web.DataReader('JPM', 'iex', start, end)
MS = web.DataReader('MS', 'iex', start, end)
WFC = web.DataReader('WFC', 'iex', start, end)

#Concatinating the data and defining hierarchy of index-level

tickers = ['BAC','C','GS','JPM','MS','WFC']
bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC], axis = 1,keys=tickers)
bank_stocks.columns.names = ['Bank Ticker','Stock Info']


#-------------------------------------------------

#Exploratory Data Analysis (EDA)

#-------------------------------------------------

bank_stocks.xs(key='close',axis=1,level='Stock Info').max()

returns = pd.DataFrame()
for tick in tickers:
    returns[tick +" Return"] = bank_stocks[tick]['close'].pct_change()

#Exploring the data using Seabonr's pairplot
sns.pairplot(returns[1:])

#Exploring the worst and best return dates
returns.idxmax()

#Interestingly 4 banks share the worst percentage-return on 2016-06-24
returns.idxmin() 

#Risk-analysis using SD of percentage-returns
returns.std()
returns.loc['2016-01-01':'2016-12-31'].std()

#Creatin distribution-plots for selected tickers, bank stocks plot and heatmap
sns.distplot(returns.loc['2016-01-01':'2016-12-31']['MS Return'],color='green',bins=100)
sns.distplot(returns.loc['2016-01-01':'2016-12-31']['WFC Return'], color='red',bins=100)

bank_stocks.xs(key='close',axis=1, level='Stock Info').plot()


sns.heatmap(bank_stocks.xs(key='close',axis=1,level='Stock Info').corr(),annot=True)