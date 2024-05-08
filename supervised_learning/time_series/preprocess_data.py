#!/usr/bin/env python3
"""
    Module to preprocess database :
    forecast analysis of the Bitcoins
"""

import pandas as pd
import plotly.graph_objs as go
from plotly.offline import iplot
import os


def plot_time_series(data):
    """
        Plot the 'Close' column of a DataFrame against its index
    :param data: dataframe pandas

    """
    # create trace for each column
    trace1 = go.Scatter(
        x=data.index,
        y=data['Open'].astype(float),
        mode='lines',
        name='Open'
    )
    trace2 = go.Scatter(
        x=data.index,
        y=data['High'].astype(float),
        mode='lines',
        name='High'
    )
    trace3 = go.Scatter(
        x=data.index,
        y=data['Low'].astype(float),
        mode='lines',
        name='Low'
    )
    trace4 = go.Scatter(
        x=data.index,
        y=data['Close'].astype(float),
        mode='lines',
        name='Close'
    )

    # layout
    layout = dict(
        title='Historical Bitcoin Price',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(count=6, label='6m', step='month', stepmode='backward'),
                    dict(count=12, label='1y', step='month', stepmode='backward'),
                    dict(count=36, label='3y', step='month', stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(visible=True),
            type='date'
        ))

    # create graph
    dataplot = [trace1, trace2, trace3, trace4]
    fig = dict(data=dataplot, layout=layout)

    iplot(fig)


def preprocess_data(path_file1, path_file2):
    """
        Complete preprocess data :
            * load,
            * conversion Timestamp in datetime
            * remove
            * fill NaN first dataset with second
            * fill forwards value of Open, High, Low and Close
    :param path_file1: first dataset
    :param path_file2: second dataset
    :return: preprocessed data
    """
    # existing file ?
    if not os.path.isfile(path_file1):
        raise FileNotFoundError(f"File {path_file1} doesn't exist.")
    if not os.path.isfile(path_file2):
        raise FileNotFoundError(f"File {path_file2} doesn't exist.")

    # load data
    print(f"Load data from {path_file1} and {path_file2}")
    df1 = pd.read_csv(path_file1)
    df2 = pd.read_csv(path_file2)

    # convert Timestamp
    df1 = df1.set_index(pd.to_datetime(df1['Timestamp'], unit='s'))
    df1 = df1.drop('Timestamp', axis=1)
    df2 = df2.set_index(pd.to_datetime(df2['Timestamp'], unit='s'))
    df2 = df2.drop('Timestamp', axis=1)

    # remove unused column
    del_col = ['Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']
    df1_clean = df1.drop(columns=del_col)
    df2_clean = df2.drop(columns=del_col)

    # use value in dataset2 to implement missing value of bitstamp
    combined_df = df1_clean.combine_first(df2_clean)

    # filter data after 2017
    combined_df2017 = combined_df[combined_df.index >= pd.Timestamp(2017, 1, 1)]

    print("before", combined_df2017.isna().sum())
    # fix missing value for Open, high, low close column : continuous timeseries
    combined_df2017['Open'] = combined_df2017['Open'].fillna(method='ffill')
    combined_df2017['High'] = combined_df2017['High'].fillna(method='ffill')
    combined_df2017['Low'] = combined_df2017['Low'].fillna(method='ffill')
    combined_df2017['Close'] = combined_df2017['Close'].fillna(method='ffill')

    print("after",combined_df2017.isna().sum())

    # save dataset
    combined_df2017.to_csv('preprocess_data.csv', index=False)

    plot_time_series(combined_df2017)

    return combined_df2017


if __name__ == "__main__":
    preprocessed_data = preprocess_data("bitstamp.csv", "coinbase.csv")
