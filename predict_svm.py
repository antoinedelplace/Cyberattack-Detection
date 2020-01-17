# -*- coding: utf-8 -*-
# Author: Antoine DELPLACE
# Last update: 17/01/2020
"""
Use Support Vector Machine to predict which flow is a malware.
Tune different hyperparameters: gamma_scale, degree, regularization penalty
Try a kernel approximation with l2 hinge

Parameters
----------
data_window.h5         : extracted data from preprocessing1.py
data_window3.h5        : extracted data from preprocessing2.py
data_window_labels.npy : label numpy array from preprocessing1.py

Return
----------
Print and Plot the results of the hyperparameter tuning (precision, recall and f1)
Print train and test accuracy, precison, recall, f1 and support for the kernel approximation
"""

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import h5py

from sklearn import model_selection, feature_selection, kernel_approximation, ensemble, linear_model, metrics

print("Import data")

X = pd.read_hdf('data_window.h5', key='data')
X.reset_index(drop=True, inplace=True)

X2 = pd.read_hdf('data_window3.h5', key='data')
X2.reset_index(drop=True, inplace=True)

X = X.join(X2)

X.drop('window_id', axis=1, inplace=True)

y = X['Label_<lambda>']
X.drop('Label_<lambda>', axis=1, inplace=True)

labels = np.load("data_window_labels.npy")

print(X.columns.values)
print(labels)
print(np.where(labels == 'flow=From-Botne')[0][0])

y_bin6 = y==np.where(labels == 'flow=From-Botne')[0][0]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_bin6, test_size=0.33, random_state=123456)

print("y", np.unique(y, return_counts=True))
print("y_train", np.unique(y_train, return_counts=True))
print("y_test", np.unique(y_test, return_counts=True))

## Train
def apply_svm_cross_validation(X, y, svc_args={'loss':'hinge', 'penalty':'elasticnet', 'max_iter':1000, 'alpha':1e-9, 'tol':1e-3, 'random_state':123456, 'class_weight':None}, kernel_args={'kernel':'rbf', 'gamma':None, 'degree':None, 'n_components':100, 'random_state':123456}):
    #print("kernel_approx")
    #feature_map_nystroem = kernel_approximation.Nystroem(**kernel_args)
    #feature_map_nystroem.fit(X)
    #X_new = feature_map_nystroem.transform(X)

    print("SVM")
    clf = linear_model.SGDClassifier(**svc_args)
    cv = model_selection.ShuffleSplit(n_splits=10, test_size=0.1, random_state=123456)
    scores = model_selection.cross_validate(clf, X, y, cv=cv, scoring=['precision', 'recall', 'f1'], return_train_score=True)
    print(scores)
    return [np.mean(scores['test_precision']), np.mean(scores['test_recall']), np.mean(scores['test_f1'])]


gamma_scale = 1/X_train.shape[1]
print("gamma_scale=", gamma_scale)
tab_gamma = np.concatenate((np.linspace(0.001, 0.04, 10), [gamma_scale]))
print(tab_gamma)

tab_score = np.array([apply_svm_cross_validation(X_train, y_train, kernel_args={'kernel':'rbf', 'gamma':gamma, 'degree':None, 'n_components':200, 'random_state':123456}) for gamma in tab_gamma])
print(tab_score)

plt.plot(tab_gamma, tab_score[:, 0])
plt.plot(tab_gamma, tab_score[:, 1])
plt.plot(tab_gamma, tab_score[:, 2])
plt.legend(["test_precision", "test_recall", "test_f1"])
plt.xlabel("Gamma")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("cross_validation_svm_gamma.pdf", format="pdf")
plt.show()


tab_degree = np.linspace(2, 4, 3)
print(tab_degree)

tab_score = np.array([apply_svm_cross_validation(X_train, y_train, kernel_args={'kernel':'poly', 'gamma':None, 'degree':degree, 'n_components':200, 'random_state':123456}) for degree in tab_degree])
print(tab_score)

plt.plot(tab_degree, tab_score[:, 0])
plt.plot(tab_degree, tab_score[:, 1])
plt.plot(tab_degree, tab_score[:, 2])
plt.legend(["test_precision", "test_recall", "test_f1"])
plt.xlabel("Degree")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("cross_validation_svm_degree.pdf", format="pdf")
plt.show()


tab_penalty = ['l1', 'l2', 'elasticnet']
print(tab_penalty)

tab_score = np.array([apply_svm_cross_validation(X_train, y_train, {'loss':'hinge', 'penalty':penalty, 'max_iter':1000, 'alpha':1e-9, 'tol':1e-3, 'random_state':123456, 'class_weight':None}) for penalty in tab_penalty])
print(tab_score)

plt.plot(tab_penalty, tab_score[:, 0])
plt.plot(tab_penalty, tab_score[:, 1])
plt.plot(tab_penalty, tab_score[:, 2])
plt.legend(["test_precision", "test_recall", "test_f1"])
plt.xlabel("Regularization Penalty")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("cross_validation_svm_penalty.pdf", format="pdf")
plt.show()


feature_map_nystroem = kernel_approximation.Nystroem(kernel='poly', gamma=None, degree=2, n_components=200, random_state=123456)
feature_map_nystroem.fit(X_train)
X_train_new = feature_map_nystroem.transform(X_train)
X_test_new = feature_map_nystroem.transform(X_test)

clf = linear_model.SGDClassifier(loss='hinge', penalty='l2', max_iter=1000, alpha=1e-9, tol=1e-3, random_state=123456, class_weight=None, verbose=1)
clf.fit(X_train_new, y_train)

print("Train")
y_pred_train = clf.predict(X_train_new)
print("accuracy score = ", metrics.balanced_accuracy_score(y_train, y_pred_train))
precision, recall, fbeta_score, support = metrics.precision_recall_fscore_support(y_train, y_pred_train)
print("precision = ", precision[1])
print("recall = ", recall[1])
print("fbeta_score = ", fbeta_score[1])
print("support = ", support[1])

print("Test")
y_pred_test = clf.predict(X_test_new)
print("accuracy score = ", metrics.balanced_accuracy_score(y_test, y_pred_test))
precision, recall, fbeta_score, support = metrics.precision_recall_fscore_support(y_test, y_pred_test)
print("precision = ", precision[1])
print("recall = ", recall[1])
print("fbeta_score = ", fbeta_score[1])
print("support = ", support[1])

