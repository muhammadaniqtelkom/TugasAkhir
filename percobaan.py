# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 16:08:52 2022

@author: mBergudeng
"""
from libraries import *
from datetime import datetime


# =============================================================================
# DATA_FOLDER = 'data/'
# filename = 'normal4.csv'
# judul ="emd_normal_4_soft.csv" 
# n_times = 1000
# 
# 
# 
# 
# 
# df = pd.read_csv(DATA_FOLDER + filename)
# 
# signal = df.iloc[1:,1][:n_times].values.astype("float64")
# #signal2 = df.iloc[1:,1].values.astype("float64")
# #x = signal;
# =============================================================================


# =============================================================================
# for wav in pywt.wavelist():
#     try:
#         filtered = wavelet_denoising(signal, wavelet=wav, level=1)
#         MSE,SNR = msesnr(signal, filtered)
#     except:
#         pass
#     
# =============================================================================
    
# =============================================================================
#     plt.figure(figsize=(10, 6))
#     plt.plot(signal, label='Raw')
#     plt.plot(filtered, label='Filtered')
#     plt.legend()
#     plt.title(f"DWT Denoising with {wav} Wavelet", size=15)
#     plt.show()
# =============================================================================
# =============================================================================
#     yable =[]
#     print('wavelet: ',wav)
#     print('SNR: ',SNR)
#     print('MSE: ',MSE)
#     yable.append(wav)
#     yable.append(SNR)
#     yable.append(MSE)
#     table.append(yable)
# =============================================================================

# =============================================================================
# # Get the default configuration for a sift
# config = emd.sift.get_config('sift')
# # Adjust the threshold for accepting an IMF
# # =============================================================================
# config['imf_opts/sd_thresh'] = 0.05
# # config['extrema_opts/method'] = 'rilling'
# # =============================================================================
# 
# imf = emd.sift.sift(signal)
# 
# 
# imf1 =imf[0:,3]
# signal = signal
# plt.plot(signal, label='Filtered')
# plt.legend()
# plt.title(f"DWT Denoising with Wavelet", size=15)
# plt.show()
# =============================================================================

# =============================================================================
# config = emd.sift.get_config('sift')
# imf_opts = config['imf_opts']
# 
# # Extract first IMF from the signal
# imf1, continue_sift = emd.sift.get_next_imf(x[:, None],           **imf_opts)
# 
# # Extract second IMF from the signal with the first IMF removed
# imf2, continue_sift = emd.sift.get_next_imf(x[:, None]-imf1,      **imf_opts)
# 
# # Extract third IMF from the signal with the first and second IMFs removed
# imf3, continue_sift = emd.sift.get_next_imf(x[:, None]-imf1-imf2, **imf_opts)
# 
# # The residual is the signal component left after removing the IMFs
# residual = x[:, None] - imf1 - imf2 - imf3
# 
# # Contactenate our IMFs into one array
# imfs_manual = np.c_[imf1, imf2, imf3, residual]
# 
# entropy1 = entropy(residual, base=2)
# 
# # Visualise
# emd.plotting.plot_imfs(imfs_manual, cmap=True, scale_y=True)
# =============================================================================

# =============================================================================
# plt.plot(residual, label='Filtered')
# plt.legend()
# plt.title(f"DWT Denoising with Wavelet", size=15)
# plt.show()
# =============================================================================

# =============================================================================
# yable =[]
# print('wavelet: ',wav)
# print('SNR: ',SNR)
# print('MSE: ',MSE)
# yable.append(wav)
# yable.append(SNR)
# yable.append(MSE)
# table.append(yable)    
# =============================================================================

# =============================================================================
# filtered = wavelet_denoising(signal, wavelet='bior3.9', level=10)
# filtered2 = wavelet_denoising_soft(signal, wavelet='bior3.9', level=10)
# MSE,SNR = msesnr(signal, filtered)
# MSE2,SNR2 = msesnr(signal, filtered2)
# 
# print('SNR_hard: ',SNR)
# print('MSE_hard: ',MSE)
# print('SNR_soft: ',SNR2)
# print('MSE_soft: ',MSE2)
# =============================================================================


# =============================================================================
# imf = emd.sift.sift(x)
# residual = imf[0:, 7]
# entropy1 = entropy(residual, base=10)
# =============================================================================
# =============================================================================
# y = imf[0:,1]
# plt.figure()
# plt.subplots_adjust(hspace=0.3)
# 
# plt.subplot(311)
# plt.plot(y)
# plt.title('Signal')
# plt.xticks([])
# =============================================================================

# =============================================================================
# threshold = [0.1 ,0.2 ,0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.9]
# threshold1 = [1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9]
# denoisedata = []
# for i in threshold1:
#     try:
#         filtered = denoising_emd(signal, i, 'soft')
#         
#         
#     except:
#         pass
#     
#     MSE,SNR = msesnr(signal, filtered)
#     ytable = []
#     ytable.append(MSE)
#     ytable.append(SNR)
#     ytable.append(i)
#     denoisedata.append(ytable)
#     
#   
# 
# =============================================================================
# =============================================================================
# folder = 'fitur_DWT'
# file = 'fixedPatient1'
# folderanotasi = 'anotasi'
# n_times = 4665
# model = DecisionTreeClassifier()
# svm.SVC()
# =============================================================================
#ppg_signal, ppg_anotation = annotation_to_ppg_signal_labeled(folder,folderanotasi,file)
#ppg_signal, ppg_anotation = annotation_to_ppg_signal_labeled2(folder,folderanotasi,file,n_times)
#ppg_signal, ppg_anotation = annotation_to_ppg_signal_labeled3(folder,folderanotasi,file,n_times)
#X, y = get_ppg_features(ppg_signal, ppg_anotation, 'bior3.9')
# =============================================================================
# ppg_signal1, ppg_anotation1 = annotation_to_ppg_signal_labeled(folder,folderanotasi,'FixedPatient3')
# X, y = get_ppg_features(ppg_signal, ppg_anotation, 'bior3.9')
# =============================================================================
# =============================================================================
# signal = ppg_signal + ppg_signal1
# anotasi = ppg_anotation + ppg_anotation1
# X, y = get_ppg_features(signal, anotasi, 'bior3.9')
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20 , random_state=42)
# run_experiment(model, X_train, y_train, X_test, y_test)
# =============================================================================

#listtocsv(X,"%s.CSV"%(file))
# =============================================================================
# listtocsv(denoisedata,judul)
# =============================================================================
# =============================================================================
# labeltest = 'AF'
# X, y = get_ppg_features2(folder, folderanotasi, file)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20 , random_state=42)
# z = run_experiment(model, X_train, y_train, X_test, y_test, labeltest)
# =============================================================================

#path = 'fitur_DWT'
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


