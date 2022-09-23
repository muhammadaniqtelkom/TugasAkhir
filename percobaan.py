# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 16:08:52 2022

@author: mBergudeng
"""
from libraries import *
from datetime import datetime


DATA_FOLDER = 'data/'
filename = 'normal4.csv'
judul ="emd_normal_4_soft.csv" 
n_times = 1000





df = pd.read_csv(DATA_FOLDER + filename)

signal = df.iloc[1:,1][:n_times].values.astype("float64")
#signal2 = df.iloc[1:,1].values.astype("float64")
#x = signal;

#denoising using DWT with variant Wavlete
for wav in pywt.wavelist():
    try:
        filtered = wavelet_denoising(signal, wavelet=wav, level=1)
        #calculate SNR and MSE
        MSE,SNR = msesnr(signal, filtered)
    except:
        pass
    
#ploting for every denoised signal    
    plt.figure(figsize=(10, 6))
    plt.plot(signal, label='Raw')
    plt.plot(filtered, label='Filtered')
    plt.legend()
    plt.title(f"DWT Denoising with {wav} Wavelet", size=15)
    plt.show()
    
    
#save to array and export to csv 
    judul = wav   
    yable =[]
    print('wavelet: ',wav)
    print('SNR: ',SNR)
    print('MSE: ',MSE)
    yable.append(wav)
    yable.append(SNR)
    yable.append(MSE)
    table.append(yable)

listtocsv(denoisedata,"EMDdenoisedfile")



#Denoising using EMD with variant Threshold
threshold = [0.1 ,0.2 ,0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.9]
threshold1 = [1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9]
denoisedata = []
for i in threshold1:
    try:
        filtered = denoising_emd(signal, i, 'soft')
        
        
    except:
        pass
    #calculating SNR and MSE
    MSE,SNR = msesnr(signal, filtered)
    
    #save to table and export to. CSV
    ytable = []
    ytable.append(MSE)
    ytable.append(SNR)
    ytable.append(i)
    denoisedata.append(ytable)
    
  
listtocsv(denoisedata,"EMDdenoisedfile")

#import signal, labeling, and cutting
folder = 'fitur_DWT'
file = 'fixedPatient1'
folderanotasi = 'anotasi'
n_times = 4665
# =============================================================================
# model = DecisionTreeClassifier()
# svm.SVC()
# =============================================================================
ppg_signal, ppg_anotation = annotation_to_ppg_signal_labeled(folder,folderanotasi,file)#labeling raw signal
ppg_signal, ppg_anotation = annotation_to_ppg_signal_labeled2(folder,folderanotasi,file,n_times)#denoising sinal using DWT and do labeling
ppg_signal, ppg_anotation = annotation_to_ppg_signal_labeled3(folder,folderanotasi,file,n_times)#denoising sinal using EMD and do labeling
X, y = get_ppg_features(ppg_signal, ppg_anotation, 'bior3.9')#feature Extraction


#feature Extraction from specific file
ppg_signal1, ppg_anotation1 = annotation_to_ppg_signal_labeled(folder,folderanotasi,'FixedPatient3')
X, y = get_ppg_features(ppg_signal, ppg_anotation, 'bior3.9')

#export feature to .CSV
listtocsv(X,"%s.CSV"%(file))

#classification test using all data from feature extraction file.
#path = 'fitur_EMD'
path = 'fitur_asli'
all_files = glob.glob(path + "/*.csv")
anotasipath = 'anotasi'
all_anotasi = glob.glob(anotasipath + "/*.csv")
model = DecisionTreeClassifier()
model1 = svm.SVC(kernel='linear')
li =[]
li1 = []
# =============================================================================
# df_list = (pd.read_csv(file) for file in all_files)
# 
# # Concatenate all DataFrames
# big_df   = pd.concat(df_list)
# =============================================================================

#data_all = pd.concat((pd.read_csv(i) for i in all_files)).reset_index(drop = True)
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=None)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)

for filename1 in all_anotasi:
    df1 = pd.read_csv(filename1)
    ss = df1["event"].values.tolist()
    li1 = li1 + ss


labeltest = "pvc"
X_train, X_test, y_train, y_test = train_test_split(frame, li1, test_size=0.20 , random_state=42)
z = run_experiment(model, X_train, y_train, X_test, y_test, labeltest)
