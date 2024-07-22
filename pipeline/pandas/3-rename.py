#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbase.csv', ',')

# rename column Timestamp to Datetime
df.rename(columns={'Timestamp': 'Datetime'}, inplace=True)

# convert timestamp values to datatime values
df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

# display only the Datetime and Close columns
df = df[['Datetime', 'Close']]

print(df.tail())
