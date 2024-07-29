#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbase.csv', ',')

# removed column 'Weighted_Price'
df.drop(columns=['Weighted_Price'], inplace=True)

# rename Timestamp to Date
df.rename(columns={'Timestamp': 'Date'}, inplace=True)

# convert in date values
df['Date'] = pd.to_datetime(df['Date'], unit='s')
df['Date'] = df['Date'].dt.to_period('d')
# only plot from 2017 and beyond
df = df.loc[df['Date'] >= "2017-01-01"]

# index data frame on Date
df = df.set_index('Date')

# missing values in Close : set to previous
df['Close'] = df['Close'].ffill()

# missing value for High, Low, Open : close value
df[['High', 'Low', 'Open']] = (
    df[['High', 'Low', 'Open']].fillna(df['Close']))

# missing value set to 0
df[['Volume_(BTC)', 'Volume_(Currency)']] = (
    df[['Volume_(BTC)', 'Volume_(Currency)']].fillna(0))

# daily group
df = df.resample('D').agg({
    'Open': 'mean',
    'High': 'max',
    'Low': 'min',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})


df = df[df.index.year >= 2017]
plt.figure(figsize=(12, 6))
df_plot = pd.DataFrame()
df_plot['Open'] = df['Open'].resample('d').mean()
df_plot['High'] = df['High'].resample('d').max()
df_plot['Low'] = df['Low'].resample('d').min()
df_plot['Close'] = df['Close'].resample('d').mean()
df_plot['Volume_(BTC)'] = df['Volume_(BTC)'].resample('d').sum()
df_plot['Volume_(Currency)'] = df['Volume_(Currency)'].resample('d').sum()

df_plot.plot(x_compat=True)
plt.show()