#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbase.csv', ',')

df = df.T
df = df.sort_index(axis=1, ascending=False)

print(df.tail(8))
