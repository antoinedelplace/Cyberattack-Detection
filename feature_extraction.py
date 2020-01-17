# -*- coding: utf-8 -*-
# Author: Antoine DELPLACE
# Last update: 17/01/2020
"""
Use different embedded methods to extract relevant features :
- Lasso and Ridge Logistic Regression
- Support Vector Machine (SVM) method with Recursive Feature Elimination (RFE)

Parameters
----------
data_window.h5         : extracted data from preprocessing1.py
data_window3.h5        : extracted data from preprocessing2.py
data_window_labels.npy : label numpy array from preprocessing1.py

Return
----------
Print the results of the different methods (precision, recall, f1)
Plot the graphs of the different extractions
"""

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import h5py

from sklearn import model_selection, feature_selection, linear_model, metrics

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

#print(X)
#print(y)
print(X.columns.values)
print(labels)

y_bin6 = y==6
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_bin6, test_size=0.33, random_state=123456)
#y_train_bin6 = y_train==6
#y_test_bin6 = y_test==6

print("y", np.unique(y, return_counts=True))
print("y_train", np.unique(y_train, return_counts=True))
print("y_test", np.unique(y_test, return_counts=True))

## Embedded Method
print("Logistic Regression")

clf = linear_model.LogisticRegression(penalty='l2', C=1.0, random_state=123456, multi_class="auto", class_weight=None, solver="lbfgs", max_iter=1000, verbose=1)
clf.fit(X_train, y_train)
#print(clf.classes_)
print(clf.coef_)
print(clf.intercept_)

y_pred = clf.predict(X_test)
#y_pred_bin6 = y_pred==6
#print(clf.predict_proba(X_test))
print("accuracy score = ", metrics.balanced_accuracy_score(y_test, y_pred))
precision, recall, fbeta_score, support = metrics.precision_recall_fscore_support(y_test, y_pred)
print("precision = ", precision[1])
print("recall = ", recall[1])
print("fbeta_score = ", fbeta_score[1])
print("support = ", support[1])

clf = linear_model.LogisticRegression(penalty='l2', C=1.0, random_state=123456, multi_class="auto", class_weight='balanced', solver="lbfgs", max_iter=1000, verbose=1)
clf.fit(X_train, y_train)
#print(clf.classes_)
print(clf.coef_)
print(clf.intercept_)

y_pred = clf.predict(X_test)
#y_pred_bin6 = y_pred==6
#print(clf.predict_proba(X_test))
print("accuracy score = ", metrics.balanced_accuracy_score(y_test, y_pred))
precision, recall, fbeta_score, support = metrics.precision_recall_fscore_support(y_test, y_pred)
print("precision = ", precision[1])
print("recall = ", recall[1])
print("fbeta_score = ", fbeta_score[1])
print("support = ", support[1])

clf = linear_model.LogisticRegression(penalty='l2', C=1.0, random_state=123456, multi_class="auto", class_weight={0:0.5, 1:0.5}, solver="lbfgs", max_iter=1000, verbose=1)
clf.fit(X_train, y_train)
#print(clf.classes_)
print(clf.coef_)
print(clf.intercept_)

y_pred = clf.predict(X_test)
#y_pred_bin6 = y_pred==6
#print(clf.predict_proba(X_test))
print("accuracy score = ", metrics.balanced_accuracy_score(y_test, y_pred))
precision, recall, fbeta_score, support = metrics.precision_recall_fscore_support(y_test, y_pred)
print("precision = ", precision[1])
print("recall = ", recall[1])
print("fbeta_score = ", fbeta_score[1])
print("support = ", support[1])

#main problems:
#with class_weight='balanced', super high recall but very low precision
#without, high precision but very low recall
#accuracy score not a good metric (even balanced_accuracy)

print("Logistic Regression Cross Validation")

def apply_logreg_cross_validation(X, y, svc_args={'penalty':'l2', 'C':1.0, 'random_state':123456, 'multi_class':"auto", 'class_weight':None, 'solver':"lbfgs", 'max_iter':1000, 'verbose':1}):
    clf = linear_model.LogisticRegression(**svc_args)
    cv = model_selection.ShuffleSplit(n_splits=10, test_size=0.1, random_state=123456)
    scores = model_selection.cross_validate(clf, X, y, cv=cv, scoring=['precision', 'recall', 'f1'], return_train_score=True)
    print(scores)
    return [np.mean(scores['test_precision']), np.mean(scores['test_recall']), np.mean(scores['test_f1'])]

tab_class_weight = np.linspace(0, 0.1, 10)
print(tab_class_weight)

tab_score = np.array([apply_logreg_cross_validation(X_train, y_train, {'penalty':'l2', 'C':1.0, 'random_state':123456, 'multi_class':"auto", 'class_weight':{0:w, 1:1-w}, 'solver':"lbfgs", 'max_iter':1000, 'verbose':0}) for w in tab_class_weight])
print(tab_score)

plt.plot(tab_class_weight, tab_score[:, 0])
plt.plot(tab_class_weight, tab_score[:, 1])
plt.plot(tab_class_weight, tab_score[:, 2])
plt.legend(["test_precision", "test_recall", "test_f1"])
plt.xlabel("Botnet class weight")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("cross_validation_class_weight.pdf", format="pdf")
plt.show()

# Results: class_weight_best = 0.044

def apply_logreg_cross_validation_coeff(X, y, svc_args={'penalty':'l2', 'C':1.0, 'random_state':123456, 'multi_class':"auto", 'class_weight':None, 'solver':"lbfgs", 'max_iter':1000, 'verbose':1}):
    clf = linear_model.LogisticRegression(**svc_args)
    #cv = model_selection.ShuffleSplit(n_splits=10, test_size=0.1, random_state=123456) #for l2
    cv = model_selection.ShuffleSplit(n_splits=3, test_size=0.1, random_state=123456) #for l1
    scores = model_selection.cross_validate(clf, X, y, cv=cv, scoring=['precision', 'recall', 'f1'], return_train_score=True, return_estimator=True)
    print(scores)
    return [np.mean(scores['test_precision']), np.mean(scores['test_recall']), np.mean(scores['test_f1']), np.mean([model.coef_[0] for model in scores['estimator']], axis=0)]

tab_C = np.logspace(-2, 6, 9)
tab_logC = np.log10(tab_C)
print(tab_C)
print(tab_logC)

tab_score = np.array([apply_logreg_cross_validation_coeff(X_train, y_train, {'penalty':'l2', 'C':C, 'random_state':123456, 'multi_class':"auto", 'class_weight':{0:0.044, 1:1-0.044}, 'solver':"lbfgs", 'max_iter':1000, 'verbose':0}) for C in tab_C])
print(tab_score)

plt.plot(tab_logC, tab_score[:, 0])
plt.plot(tab_logC, tab_score[:, 1])
plt.plot(tab_logC, tab_score[:, 2])
plt.legend(["test_precision", "test_recall", "test_f1"])
plt.xlabel("log(C)")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("cross_validation_C.pdf", format="pdf")
plt.show()

matrix_coeff = np.stack(tab_score[:, 3], axis=0)
print(matrix_coeff)
print(matrix_coeff.shape)

ax = plt.subplot(111)
NUM_COLORS = matrix_coeff.shape[1]
LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
NUM_STYLES = len(LINE_STYLES)
cm = plt.get_cmap('Set1')

for i in range(0, matrix_coeff.shape[1]):
    lines = ax.plot(tab_logC, matrix_coeff[:, i])
    lines[0].set_color(cm(i//NUM_STYLES))
    lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])

plt.xlabel("log(C)")
plt.xticks(label=np.log(tab_C))

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
ax.legend(np.arange(0, matrix_coeff.shape[1]), loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig("cross_validation_C_coeff.pdf", format="pdf")
plt.show()

print(matrix_coeff)


tab_C = np.linspace(550, 1000, 10)
print(tab_C)

tab_score = np.array([apply_logreg_cross_validation_coeff(X_train, y_train, {'penalty':'l2', 'C':C, 'random_state':123456, 'multi_class':"auto", 'class_weight':{0:0.044, 1:1-0.044}, 'solver':"lbfgs", 'max_iter':1000, 'verbose':0}) for C in tab_C])
print(tab_score)

plt.plot(tab_C, tab_score[:, 0])
plt.plot(tab_C, tab_score[:, 1])
plt.plot(tab_C, tab_score[:, 2])
plt.legend(["test_precision", "test_recall", "test_f1"])
plt.xlabel("C")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("cross_validation_C.pdf", format="pdf")
plt.show()

matrix_coeff = np.stack(tab_score[:, 3], axis=0)
print(matrix_coeff)
print(matrix_coeff.shape)

ax = plt.subplot(111)
NUM_COLORS = matrix_coeff.shape[1]
LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
NUM_STYLES = len(LINE_STYLES)
cm = plt.get_cmap('Set1')

for i in range(0, matrix_coeff.shape[1]):
    lines = ax.plot(tab_C, matrix_coeff[:, i])
    lines[0].set_color(cm(i//NUM_STYLES))
    lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])

plt.xlabel("C")

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
ax.legend(np.arange(0, matrix_coeff.shape[1]), loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig("cross_validation_C_coeff.pdf", format="pdf")
plt.show()

tab_C = np.linspace(50, 1000, 20)
print(tab_C)

print(tab_score)

plt.plot(tab_C, tab_score[:, 0])
plt.plot(tab_C, tab_score[:, 1])
plt.plot(tab_C, tab_score[:, 2])
plt.legend(["test_precision", "test_recall", "test_f1"], loc='upper right', bbox_to_anchor=(1, 0.9))
plt.xlabel("C")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("cross_validation_C.pdf", format="pdf")
plt.show()

matrix_coeff = np.stack(tab_score[:, 3], axis=0)
print(matrix_coeff)
print(matrix_coeff.shape)

ax = plt.subplot(111)
NUM_COLORS = matrix_coeff.shape[1]
LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
NUM_STYLES = len(LINE_STYLES)
cm = plt.get_cmap('Set1')

for i in range(0, matrix_coeff.shape[1]):
    lines = ax.plot(tab_C, matrix_coeff[:, i])
    lines[0].set_color(cm(i//NUM_STYLES))
    lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])

plt.xlabel("C")

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
ax.legend(np.arange(0, matrix_coeff.shape[1]), loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig("cross_validation_C_coeff.pdf", format="pdf")
plt.show()


#tab_C = np.logspace(-2, 6, 9)
tab_C = [1e6]
tab_logC = np.log10(tab_C)
print(tab_C)
print(tab_logC)

tab_score = np.array([apply_logreg_cross_validation_coeff(X_train, y_train, {'penalty':'l1', 'C':C, 'random_state':123456, 'multi_class':"auto", 'class_weight':{0:0.044, 1:1-0.044}, 'solver':"liblinear", 'max_iter':1000, 'verbose':1}) for C in tab_C])
print(tab_score)

plt.plot(tab_logC, tab_score[:, 0])
plt.plot(tab_logC, tab_score[:, 1])
plt.plot(tab_logC, tab_score[:, 2])
plt.legend(["test_precision", "test_recall", "test_f1"])
plt.xlabel("log(C)")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("cross_validation_C.pdf", format="pdf")
plt.show()

matrix_coeff = np.stack(tab_score[:, 3], axis=0)
print(matrix_coeff)
print(matrix_coeff.shape)

ax = plt.subplot(111)
NUM_COLORS = matrix_coeff.shape[1]
LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
NUM_STYLES = len(LINE_STYLES)
cm = plt.get_cmap('Set1')

for i in range(0, matrix_coeff.shape[1]):
    lines = ax.plot(tab_logC, matrix_coeff[:, i])
    lines[0].set_color(cm(i//NUM_STYLES))
    lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])

plt.xlabel("log(C)")
plt.xticks(label=np.log(tab_C))

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
ax.legend(np.arange(0, matrix_coeff.shape[1]), loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig("cross_validation_C_coeff.pdf", format="pdf")
plt.show()


print("SVM method with RFE")

def extract_feature(clf, X, y):
    pass

def rfe_svm(X, y):
    clf = linear_model.SGDClassifier(loss='hinge', penalty='elasticnet', max_iter=1000, alpha=1e-9, tol=1e-3, random_state=123456, class_weight={0:0.044, 1:1-0.044})
    cv = model_selection.ShuffleSplit(n_splits=10, test_size=0.1, random_state=123456)

    nb_features = X.shape[1]
    print(nb_features)
    
    scores = model_selection.cross_validate(clf, X, y, cv=cv, scoring=['precision', 'recall', 'f1'], return_train_score=True)
    print(scores)
    
    if nb_features > 1:
        rfe = feature_selection.RFE(clf, n_features_to_select=nb_features-1, step=1)
        rfe.fit(X, y)
        output = rfe_svm(rfe.transform(X), y)
        
        output.append([nb_features, np.mean(scores['test_precision']), np.mean(scores['test_recall']), np.mean(scores['test_f1']), rfe.support_, rfe.ranking_])
        return output
    else:
        return [[nb_features, np.mean(scores['test_precision']), np.mean(scores['test_recall']), np.mean(scores['test_f1']), [True], [1]]]


results = np.array(rfe_svm(X_train, y_train))
print(results)

plt.plot(results[:, 0], results[:, 1])
plt.plot(results[:, 0], results[:, 2])
plt.plot(results[:, 0], results[:, 3])
plt.legend(["test_precision", "test_recall", "test_f1"])
plt.xlabel("Number of features")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("cross_validation_rfe.pdf", format="pdf")
plt.show()


def apply_svm_cross_validation(X, y, svc_args={'loss':'hinge', 'penalty':'elasticnet', 'max_iter':1000, 'alpha':0.001, 'tol':1e-3, 'random_state':123456, 'class_weight':None}):
    clf = linear_model.SGDClassifier(**svc_args)
    cv = model_selection.ShuffleSplit(n_splits=10, test_size=0.1, random_state=123456)
    scores = model_selection.cross_validate(clf, X, y, cv=cv, scoring=['precision', 'recall', 'f1'], return_train_score=True)
    print(scores)
    return [np.mean(scores['test_precision']), np.mean(scores['test_recall']), np.mean(scores['test_f1'])]

tab_class_weight = np.linspace(0, 0.1, 10)
print(tab_class_weight)

tab_score = np.array([apply_svm_cross_validation(X_train, y_train, {'loss':'hinge', 'penalty':'elasticnet', 'max_iter':1000, 'alpha':0.001, 'tol':1e-3, 'random_state':123456, 'class_weight':{0:w, 1:1-w}}) for w in tab_class_weight])
print(tab_score)

plt.plot(tab_class_weight, tab_score[:, 0])
plt.plot(tab_class_weight, tab_score[:, 1])
plt.plot(tab_class_weight, tab_score[:, 2])
plt.legend(["test_precision", "test_recall", "test_f1"])
plt.xlabel("Botnet class weight")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("cross_validation_class_weight_svm.pdf", format="pdf")
plt.show()


tab_C = np.logspace(-16, -8, 9)
tab_logC = np.log10(tab_C)
print(tab_C)
print(tab_logC)

tab_score = np.array([apply_svm_cross_validation(X_train, y_train, {'loss':'hinge', 'penalty':'elasticnet', 'max_iter':1000, 'alpha':C, 'tol':1e-3, 'random_state':123456, 'class_weight':{0:0.044, 1:1-0.044}}) for C in tab_C])
print(tab_score)

plt.plot(tab_logC, tab_score[:, 0])
plt.plot(tab_logC, tab_score[:, 1])
plt.plot(tab_logC, tab_score[:, 2])
plt.legend(["test_precision", "test_recall", "test_f1"])
plt.xlabel("log(alpha)")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("cross_validation_alpha.pdf", format="pdf")
plt.show()
