# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 14:36:54 2022

@author: mBergudeng
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
import scipy
import csv
import math
import os
import time
import numpy as np
import pandas as pd
import emd
import datetime as dt
import scipy.io as sio
import scipy.stats
import glob

from sklearn.tree import DecisionTreeClassifier

from sklearn import *
from sklearn.metrics import *
from sklearn.metrics import confusion_matrix
from collections import defaultdict, Counter
from scipy.fftpack import fft
from IPython.display import display
from scipy.stats import entropy
from sklearn.model_selection import train_test_split






""" DENOISING DWT """

def madev(d, axis=None):
    """ Mean absolute deviation of a signal """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

"""HARD THRESHOLD"""
def wavelet_denoising(x, wavelet, level):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    
    """ calculate Universal Threshold"""
    sigma = (1/0.6745) * madev(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    
    """ Hard Threshold"""
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')


"""SOFT THRESHOLD"""
def wavelet_denoising_soft(x, wavelet, level):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    
    """ calculate Universal Threshold"""
    sigma = (1/0.6745) * madev(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    
    """ Hard Threshold"""
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')

""""""""

"""EMD """
def denoising_emd(signal,threshold, jenis):
    imf = emd.sift.sift(signal)
    imf1 = imf[0:,0]
    denoised = pywt.threshold(imf1, threshold, jenis)
    signalfree = signal - denoised
    return signalfree


"""""SNR and MSE"""""

def msesnr(signal, filtered):
    noise = signal - filtered
    
    mse = np.mean(noise**2)
    snr1 = np.mean(filtered**2)/np.mean(noise**2)
    snr = 10 * math.log10(snr1)
    return mse,snr


""""""


"""Ekstraksi Fitur"""

""""""

def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=scipy.stats.entropy(probabilities)
    return entropy

def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values**2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]

def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]

def get_features(list_values):
    entropy = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    return [entropy] + crossings + statistics

# =============================================================================
# def get_uci_har_features(dataset, labels, waveletname):
#     uci_har_features = []
#     for signal_no in range(0, len(dataset)):
#         features = []
#         for signal_comp in range(0,dataset.shape[2]):
#             signal = dataset[signal_no, :, signal_comp]
#             list_coeff = pywt.wavedec(signal, waveletname)
#             for coeff in list_coeff:
#                 features += get_features(coeff)
#         uci_har_features.append(features)
#     X = np.array(uci_har_features)
#     Y = np.array(labels)
#     return X, Y
# =============================================================================


def get_ppg_features(dataset, labels, waveletname):
    ppg_features = []
    for signal_no in range(0, len(dataset)):
      signal = dataset[signal_no]
      list_coeff = pywt.wavedec(signal, waveletname, level=3)
      features = []
      for coeff in list_coeff:
          features += get_features(coeff)
      ppg_features.append(features)
    X = np.array(ppg_features)
    Y = np.array(labels)
    
    
    
    return X, Y

""""""
def get_ppg_features2(signal_path, annotation_path, number):
    signal_file = "%s/%s.csv" % (signal_path, number)
    anot_file = "%s/%s.csv" % (annotation_path, number)
    signal = pd.read_csv(signal_file, header=None)
    annotation = pd.read_csv(anot_file, low_memory=False)
    ppg = signal.values.tolist()
    labels = annotation["event"].values.tolist()
    X = np.array(ppg)
    Y = np.array(labels)
    
    
    
    return X, Y




""""""
def annotation_to_ppg_signal_labeled(signal_path, annotation_path, number):
    signal_file = "%s/%s.csv" % (signal_path, number)
    anot_file = "%s/%s.csv" % (annotation_path, number)
    signal = pd.read_csv(signal_file, low_memory=False).replace("-",100)
    annotation = pd.read_csv(anot_file, low_memory=False)
    ppg = signal.iloc[1:,1].values
    ppg_signal = []
    start, stop, signal_class = annotation["Indx1"].values, annotation["Indx2"].values, annotation["event"].values.tolist()
    for i in range(len(annotation)):
            ppg_cut = ppg[start[i]:stop[i]].astype("float64").tolist()
            ppg_signal.append(ppg_cut)
    return ppg_signal, signal_class



""""""
def annotation_to_ppg_signal_labeled2(signal_path, annotation_path, number, n_times):
    signal_file = "%s/%s.csv" % (signal_path, number)
    anot_file = "%s/%s.csv" % (annotation_path, number)
    signal = pd.read_csv(signal_file, low_memory=False).replace("-",100)
    annotation = pd.read_csv(anot_file, low_memory=False)
    ppg = signal.iloc[1:,1][:n_times].values
    filterred = wavelet_denoising(ppg, "bior3.9", 1)
    ppg_signal = []
    start, stop, signal_class = annotation["Indx1"].values, annotation["Indx2"].values, annotation["event"].values.tolist()
    for i in range(len(annotation)):
            ppg_cut = filterred[start[i]:stop[i]].astype("float64").tolist()
            ppg_signal.append(ppg_cut)
    return ppg_signal, signal_class

""""""
def annotation_to_ppg_signal_labeled3(signal_path, annotation_path, number, n_times):
    signal_file = "%s/%s.csv" % (signal_path, number)
    anot_file = "%s/%s.csv" % (annotation_path, number)
    signal = pd.read_csv(signal_file, low_memory=False).replace("-",100)
    annotation = pd.read_csv(anot_file, low_memory=False)
    ppg = signal.iloc[1:,1][:n_times].values.astype("float64")
    filterred = denoising_emd(ppg, 9, "soft")
    ppg_signal = []
    start, stop, signal_class = annotation["Indx1"].values, annotation["Indx2"].values, annotation["event"].values.tolist()
    for i in range(len(annotation)):
            ppg_cut = filterred[start[i]:stop[i]].astype("float64").tolist()
            ppg_signal.append(ppg_cut)
    return ppg_signal, signal_class

""""""
def run_experiment(model, x_train, y_train, x_test, y_test, label):
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    
    #plot_confusion_matrix(model, x_test,y_test, cmap='GnBu')
    ConfusionMatrixDisplay.from_estimator(model, x_test, y_test)
    plt.show()
# =============================================================================
#     precision = precision_score(y_test, y_pred, pos_label= 'AF')
#     recall = recall_score(y_test, y_pred, pos_label= 'AF')
#     f1 = f1_score(y_test, y_pred, pos_label= 'AF')
# =============================================================================
# =============================================================================
#     accuracy = accuracy_score(y_test, y_pred)
#     print('Precision: %.3f' % precision_score(y_test, y_pred, average='macro'))
#     print('Recall: %.3f' % recall_score(y_test, y_pred,  average='macro'))
#     return  accuracy
# =============================================================================
    print('Precision: %.3f' % precision_score(y_test, y_pred, average='macro'))
    print('Recall: %.3f' % recall_score(y_test, y_pred,  average='macro'))
    print('F1: %.3f' % f1_score(y_test, y_pred, average='macro'))
    print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))

""""""
def get_train_test(df, y_col, x_cols, ratio):
    
    """ 
    This method transforms a dataframe into a train and test set, for this you need to specify:
    1. the ratio train : test (usually 0.7)
    2. the column with the Y_values
    
    """
    mask = np.random.rand(len(df)) < ratio
    df_train = df[mask]
    df_test = df[~mask]
       
    Y_train = df_train[y_col].values
    Y_test = df_test[y_col].values
    X_train = df_train[x_cols].values
    X_test = df_test[x_cols].values
    return df_train, df_test, X_train, Y_train, X_test, Y_test


"""Convert To PDF"""
def listtocsv(data,judul):
    np.savetxt(judul, data, delimiter=", ", fmt="% s")
    return

""""""
