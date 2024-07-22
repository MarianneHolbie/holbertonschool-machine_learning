#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df1 = from_file('coinbase.csv', ',')
df2 = from_file('bitstamp.csv', ',')

# Include all timestamps  up to (including)  1417411920
df2 = df2.loc[df2['Timestamp'] <= 1417411920]

# reset index
df1 = df1.set_index('Timestamp')
df2 = df2.set_index('Timestamp')

# add Keys bitstamp, coinbase and concat
df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])

print(df)
