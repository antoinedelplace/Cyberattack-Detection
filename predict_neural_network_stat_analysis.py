# -*- coding: utf-8 -*-
# Author: Antoine DELPLACE
# Last update: 17/01/2020
"""
Perform a statistic analysis of the Neural network classifier.

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
import time

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, AveragePooling2D, Dense, Dropout, Flatten, Lambda, MaxPool2D, Conv2DTranspose, UpSampling2D, Concatenate, Add
from tensorflow.keras import regularizers, optimizers
from keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn import model_selection, feature_selection, utils, ensemble, linear_model, metrics

print("Import data")

X = pd.read_hdf('data_window_botnet7.h5', key='data')
X.reset_index(drop=True, inplace=True)

X2 = pd.read_hdf('data_window3_botnet7.h5', key='data')
X2.reset_index(drop=True, inplace=True)

X = X.join(X2)

X.drop('window_id', axis=1, inplace=True)

y = X['Label_<lambda>']
X.drop('Label_<lambda>', axis=1, inplace=True)

labels = np.load("data_window_botnet7_labels.npy")

print(X.columns.values)
print(labels)
print(np.where(labels == 'flow=From-Botne')[0][0])

y_bin6 = y==np.where(labels == 'flow=From-Botne')[0][0]
print("y", np.unique(y, return_counts=True))

## NN
filename_weights = "model.h5"

def fprecision(y_true, y_pred):	
    """Precision metric.	
    Only computes a batch-wise average of precision. Computes the precision, a
    metric for multi-label classification of how many selected items are
    relevant.
    """	
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))	
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))	
    precision = true_positives / (predicted_positives + K.epsilon())	
    return precision

def frecall(y_true, y_pred):	
    """Recall metric.	
    Only computes a batch-wise average of recall. Computes the recall, a metric
    for multi-label classification of how many relevant items are selected.	
    """	
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))	
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))	
    recall = true_positives / (possible_positives + K.epsilon())	
    return recall

def ff1_score(y_true, y_pred):
    """Computes the F1 Score
    Only computes a batch-wise average of recall. Computes the recall, a metric
    for multi-label classification of how many relevant items are selected.	
    """
    p = fprecision(y_true, y_pred)
    r = frecall(y_true, y_pred)
    return (2 * p * r) / (p + r + K.epsilon())

def get_model(inputs, dropout=0.5, batchnorm=True):
    x = Dense(256, input_shape=(22,))(inputs)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout)(x)

    x = Dense(128, input_shape=(256,))(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout)(x)

    x = Dense(1, input_shape=(128,))(x)
    outputs = Activation("sigmoid")(x)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

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

    print(i)
    print("y_train", np.unique(y_train, return_counts=True))
    print("y_test", np.unique(y_test, return_counts=True))

    inputs = Input((22,), name='input')
    model = get_model(inputs, dropout=0, batchnorm=1)

    callbacks = [
        ModelCheckpoint(filename_weights, verbose=1, save_best_only=True, save_weights_only=True)
    ]

    model.compile(optimizer=optimizers.Adam(lr=1e-3), loss=["binary_crossentropy"], metrics=[fprecision, frecall, ff1_score])
    #model.summary()

    tps = time.time()
    results = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_split=0.15, shuffle=True, class_weight=None, verbose=0, callbacks=callbacks)
    print("Execution time = ", time.time()-tps)

    model.load_weights(filename_weights)

    y_pred_train = model.predict(X_train, batch_size=32, verbose=0)
    y_pred_train_bin = (y_pred_train > 0.5).astype(np.uint8)
    precision, recall, fbeta_score, support = metrics.precision_recall_fscore_support(y_train, y_pred_train_bin)
    tab_train_precision[i] = precision[1]
    tab_train_recall[i] = recall[1]
    tab_train_fbeta_score[i] = fbeta_score[1]

    y_pred_test = model.predict(X_test, batch_size=32, verbose=0)
    y_pred_test_bin = (y_pred_test > 0.5).astype(np.uint8)
    precision, recall, fbeta_score, support = metrics.precision_recall_fscore_support(y_test, y_pred_test_bin)
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
