#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbase.csv', ',')

df = df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[:: 60]

print(df.tail())
