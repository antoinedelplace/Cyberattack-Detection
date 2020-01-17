# -*- coding: utf-8 -*-
# Author: Antoine DELPLACE
# Last update: 17/01/2020
"""
Pre-processing program to extract window-related data from Netflow files

Parameters
----------
window_width  : window width in seconds
window_stride : window stride in seconds
data          : pandas DataFrame of the Netflow file

Return
----------
Create 3 output files:
- data_window_botnetx.h5         : DataFrame with the extracted data: Sport, DstAddr, Dport,
                                   Dur (sum, mean, std, max, median), TotBytes (sum, mean, std, max, median),
                                   SrcBytes (sum, mean, std, max, median)
- data_window_botnetx_id.npy     : Numpy array containing SrcAddr
- data_window_botnetx_labels.npy : Numpy array containing Label
"""

import pandas as pd
import numpy as np
import datetime
import h5py

from scipy.stats import mode

window_width = 120 # seconds
window_stride = 60 # seconds

print("Import data")
data = pd.read_csv("capture20110812_3.binetflow")
#with pd.option_context('display.max_rows', None, 'display.max_columns', 15):
#    print(data.shape)
#    print(data.head())
#    print(data.dtypes)

print("Preprocessing")
def normalize_column(dt, column):
    mean = dt[column].mean()
    std = dt[column].std()
    print(mean, std)

    dt[column] = (dt[column]-mean) / std

data['StartTime'] = pd.to_datetime(data['StartTime']).astype(np.int64)*1e-9
datetime_start = data['StartTime'].min()

data['Window_lower'] = (data['StartTime']-datetime_start-window_width)/window_stride+1
data['Window_lower'].clip(lower=0, inplace=True)
data['Window_upper_excl'] = (data['StartTime']-datetime_start)/window_stride+1
data = data.astype({"Window_lower": int, "Window_upper_excl": int})
data.drop('StartTime', axis=1, inplace=True)

data['Label'], labels = pd.factorize(data['Label'].str.slice(0, 15))
#print(data.dtypes)

X = pd.DataFrame()
nb_windows = data['Window_upper_excl'].max()
print(nb_windows)

for i in range(0, nb_windows):
    gb = data.loc[(data['Window_lower'] <= i) & (data['Window_upper_excl'] > i)].groupby('SrcAddr')
    X = X.append(gb.size().to_frame(name='counts').join(gb.agg({'Sport':'nunique', 
                                                       'DstAddr':'nunique', 
                                                       'Dport':'nunique', 
                                                       'Dur':['sum', 'mean', 'std', 'max', 'median'],
                                                       'TotBytes':['sum', 'mean', 'std', 'max', 'median'],
                                                       'SrcBytes':['sum', 'mean', 'std', 'max', 'median'],
                                                       'Label':lambda x: mode(x)[0]})).reset_index().assign(window_id=i))
    print(X.shape)

del(data)

X.columns = ["_".join(x) if isinstance(x, tuple) else x for x in X.columns.ravel()]
#print(X.columns.values)

# std can be Nan if only one element
X.fillna(-1, inplace=True)

#print(X.columns.values)
columns_to_normalize = list(X.columns.values)
columns_to_normalize.remove('SrcAddr')
columns_to_normalize.remove('Label_<lambda>')
columns_to_normalize.remove('window_id')

normalize_column(X, columns_to_normalize)

with pd.option_context('display.max_rows', 10, 'display.max_columns', 22):
    print(X.shape)
    print(X)
    print(X.dtypes)
    
#with pd.option_context('display.max_rows', 10, 'display.max_columns', 20):
#    print(X.loc[X['Label'] != 0])

X.drop('SrcAddr', axis=1).to_hdf('data_window_botnet3.h5', key="data", mode="w")
np.save("data_window_botnet3_id.npy", X['SrcAddr'])
np.save("data_window_botnet3_labels.npy", labels)
