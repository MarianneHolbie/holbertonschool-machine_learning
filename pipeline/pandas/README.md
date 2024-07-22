# Pandas

Pandas is a powerful data manipulation library for Python. It provides fast, flexible, and expressive data structures designed to make working with structured (tabular, multidimensional, potentially heterogeneous) and time series data both easy and intuitive.

## Installation

To install pandas, use the following command:

```bash
pip install pandas
```

## Tasks

| Task                                        | Description                                                                                                              |
|---------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| [From Numpy](./0-from_numpy.py)             | Function `from_numpy` that creates a `pd.DataFrame` from a `np.ndarray`                                                  |
| [From Dictionary](./1-from_dictionary.py)   | Function `from_dictionary` that creates a `pd.DataFrame` from a dictionary                                               |
| [From File](./2-from_file.py)               | Function `from_file` that loads data from a file as a `pd.DataFrame`                                                     |
| [Rename](./3-rename.py)                     | Script that renames the columns, convert timestamp values to datatime values and display only Datetime and Close columns |
| [To Numpy](./4-array.py)                    | script that take last 10 rows of the columns High and Close, and convert them into a `np.ndarray`                        |
| [Slice](./5-slice.py)                       | Script that slices the pd.DataFrame along the columns High, Low, Close, and Volume for every 60th row                    |
| [Flip it and Switch it](./6-flip_switch.py) | Script that transposes the pd.DataFrame and swaps the columns and rows                                                   |
| [Sort](./7-high_low.py)                     | Script that sorts the pd.DataFrame by the High Price in descending order                                                 |
| [Prune](./8-prune.py)                       | Script that remove the entries in the pd.DataFrame where `Close` is `NaN`                                                |                                                   
| [Fill](./9-fill.py)                         | Script that fills in the missing data points in the pd.DataFrame                                                         |
| [Indexing](./10-index.py)                   | Script that sets the index of the pd.DataFrame to the `Timestamp` column                                                 |
| [Concat](./11-concat.py)                    | Script that concatenates the pd.DataFrames of the same or different shapes                                               |
| [Hierarchy](./12-hierarchy.py)              | Based on 11-concat.py rearrange the MultiIndex levels such that timestamp is the first level                             |                            
| [Analyse](./13-analyze.py)                  | Script that calculates descriptive statistics of the pd.DataFrame except Timestamp                                       |
| [Visualize](./14-visualize.py)              | Script that plots the pd.DataFrame with labels and titles                                                                |

