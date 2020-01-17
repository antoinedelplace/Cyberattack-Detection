# -*- coding: utf-8 -*-
# Author: Antoine DELPLACE
# Last update: 17/01/2020
"""
Perform a statistic analysis of the Random Forest with a bootstrap method to predict which flow is a malware.

Parameters
----------
data_window_botnetx.h5         : extracted data from preprocessing1.py
data_window3_botnetx.h5        : extracted data from preprocessing2.py
data_window_botnetx_labels.npy : label numpy array from preprocessing1.py
nb_prediction                  : number of predictions to perform

Return
----------
Print train and test mean accuracy, precison, recall, f1
"""

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import h5py

from sklearn import model_selection, feature_selection, utils, ensemble, linear_model, metrics

print("Import data")

X = pd.read_hdf('data_window_botnet5.h5', key='data')
X.reset_index(drop=True, inplace=True)

X2 = pd.read_hdf('data_window3_botnet5.h5', key='data')
X2.reset_index(drop=True, inplace=True)

X = X.join(X2)

X.drop('window_id', axis=1, inplace=True)

y = X['Label_<lambda>']
X.drop('Label_<lambda>', axis=1, inplace=True)

labels = np.load("data_window_botnet5_labels.npy")

print(X.columns.values)
print(labels)
print(np.where(labels == 'flow=From-Botne')[0][0])

print("y", np.unique(y, return_counts=True))

y_bin6 = y==np.where(labels == 'flow=From-Botne')[0][0]

nb_prediction = 50
np.random.seed(seed=123456)
tab_seed = np.random.randint(0, 1000000000, nb_prediction)
print(tab_seed)

tab_train_precision = np.array([0.]*nb_prediction)
tab_train_recall = np.array([0.]*nb_prediction)
tab_train_fbeta_score = np.array([0.]*nb_prediction)

tab_test_precision = np.array([0.]*nb_prediction)
tab_test_recall = np.array([0.]*nb_prediction)
tab_test_fbeta_score = np.array([0.]*nb_prediction)

for i in range(0, nb_prediction):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_bin6, test_size=0.33, random_state=tab_seed[i])

    X_train_new, y_train_new = utils.resample(X_train, y_train, n_samples=X_train.shape[0]*30, random_state=tab_seed[i])

    print(i)
    print("y_train", np.unique(y_train_new, return_counts=True))
    print("y_test", np.unique(y_test, return_counts=True))

    clf = ensemble.RandomForestClassifier(n_estimators=100, random_state=tab_seed[i], verbose=0, class_weight=None)
    clf.fit(X_train_new, y_train_new)

    y_pred_train = clf.predict(X_train_new)
    precision, recall, fbeta_score, support = metrics.precision_recall_fscore_support(y_train_new, y_pred_train)
    tab_train_precision[i] = precision[1]
    tab_train_recall[i] = recall[1]
    tab_train_fbeta_score[i] = fbeta_score[1]

    y_pred_test = clf.predict(X_test)
    precision, recall, fbeta_score, support = metrics.precision_recall_fscore_support(y_test, y_pred_test)
    tab_test_precision[i] = precision[1]
    tab_test_recall[i] = recall[1]
    tab_test_fbeta_score[i] = fbeta_score[1]

print("Train")
print("precision = ", tab_train_precision.mean(), tab_train_precision.std(), tab_train_precision)
print("recall = ", tab_train_recall.mean(), tab_train_recall.std(), tab_train_recall)
print("fbeta_score = ", tab_train_fbeta_score.mean(), tab_train_fbeta_score.std(), tab_train_fbeta_score)

print("Test")
print("precision = ", tab_test_precision.mean(), tab_test_precision.std(), tab_test_precision)
print("recall = ", tab_test_recall.mean(), tab_test_recall.std(), tab_test_recall)
print("fbeta_score = ", tab_test_fbeta_score.mean(), tab_test_fbeta_score.std(), tab_test_fbeta_score)
