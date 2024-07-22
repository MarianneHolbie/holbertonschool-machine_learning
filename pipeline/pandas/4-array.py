#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbase.csv', ',')

A = df[['High', 'Close']].tail(10)

A = A.to_numpy()

print(A)
