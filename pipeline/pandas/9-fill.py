#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbase.csv', ',')

df.drop(columns=['Weighted_Price'], inplace=True)
df['Close'] = df['Close'].ffill()
df[['High', 'Low', 'Open']] = (
    df[['High', 'Low', 'Open']].fillna(df['Close']))
df[['Volume_(BTC)', 'Volume_(Currency)']] = (
    df[['Volume_(BTC)', 'Volume_(Currency)']].fillna(0))


print(df.head())
print(df.tail())
