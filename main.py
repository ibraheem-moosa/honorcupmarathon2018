import sys
import os
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.multioutput import MultiOutputRegressor
from scipy.fftpack import fft
from sklearn.metrics import mean_squared_error
import numpy as np
import random
from math import sqrt
import matplotlib.pyplot as plt

random.seed(42)
is_test = False
if len(sys.argv) > 2:
    is_test = True

data_dir = sys.argv[1]
dataset = dict()

for fname in os.listdir(data_dir):
    user_id = fname[4:fname.find("log")]
    if user_id not in dataset:
        dataset[user_id] = []
    with open(os.path.join(data_dir, fname)) as f:
        data = [l.split(",") for l in f]
        sbp = int(data[0][0])
        dbp = int(data[0][1])
        ecg = [int(val[0]) for val in data[-13000:]]
        ppg = [int(val[1]) for val in data[-13000:]]
        if len(ecg) != 13000:
            print(len(ecg))
            plt.plot(ecg)
            plt.show()
            plt.plot(ppg)
            plt.show()
        dataset[user_id].append((fname, sbp, dbp, ecg, ppg))

freqs_to_skip = 500
ecg_num_freqs = 5000
ppg_num_freqs = 5000

normalize = True
normalize = False
sbp_reg = KNeighborsRegressor(n_neighbors=1, weights='uniform', p=2)#, metric='chebyshev')
dbp_reg = KNeighborsRegressor(n_neighbors=1, weights='uniform', p=2)#, metric='chebyshev')
sbp_reg = RandomForestRegressor(n_estimators=10000, n_jobs=-1, max_features='sqrt')
dbp_reg = RandomForestRegressor(n_estimators=10000, n_jobs=-1, max_features='sqrt')


output_file = open("results.txt", "w")

sbp_mse = 0.0
dbp_mse = 0.0
for user in dataset:
    data = dataset[user]
    num_of_samples = len(data)
    num_of_cal_samples = num_of_samples // 5
    print(user, num_of_samples)
    indices = list(range(len(data))) 
    random.shuffle(indices)

    features = []
    sbps = []
    dbps = []
    fnames = []
    cal_indices = []
    tst_indices = []

    for i, sample in enumerate(data):
        fname, sbp, dbp, ecg, ppg = sample
        sbps.append(sbp)
        dbps.append(dbp)
        ecg_fft = np.absolute(fft(ecg))[freqs_to_skip:freqs_to_skip+ecg_num_freqs]
        ppg_fft = np.absolute(fft(ppg))[freqs_to_skip:freqs_to_skip+ppg_num_freqs]
        #plt.plot(ecg_fft)
        #plt.show()
        #plt.plot(ppg_fft)
        #plt.show()
        if normalize:
            ecg_fft /= np.max(ecg_fft[0])
            ppg_fft /= np.max(ppg_fft[0])
        features.append(np.hstack((ecg_fft, ppg_fft)))
        if sbp == 0 and dbp == 0:
            tst_indices.append(i)
        else:
            cal_indices.append(i)
        fnames.append(fname)

    if is_test:
        num_of_cal_samples = len(cal_indices)
        indices = cal_indices + tst_indices
    #print(outputs)

    cal_features = np.vstack([features[i] for i in indices[:num_of_cal_samples]])
    cal_sbps = np.array([sbps[i] for i in indices[:num_of_cal_samples]])
    cal_dbps = np.array([dbps[i] for i in indices[:num_of_cal_samples]])

    tst_features = np.vstack([features[i] for i in indices[num_of_cal_samples:]])
    tst_sbps = np.array([sbps[i] for i in indices[num_of_cal_samples:]])
    tst_dbps = np.array([dbps[i] for i in indices[num_of_cal_samples:]])
    
    sbp_reg.fit(cal_features, cal_sbps)
    dbp_reg.fit(cal_features, cal_dbps)

    tst_sbp_predictions = sbp_reg.predict(tst_features)
    tst_dbp_predictions = dbp_reg.predict(tst_features)
    tst_sbp_mse = mean_squared_error(tst_sbp_predictions, tst_sbps)
    tst_dbp_mse = mean_squared_error(tst_dbp_predictions, tst_dbps)

    cal_sbp_predictions = sbp_reg.predict(cal_features)
    cal_dbp_predictions = dbp_reg.predict(cal_features)
    cal_sbp_mse = mean_squared_error(cal_sbp_predictions, cal_sbps)
    cal_dbp_mse = mean_squared_error(cal_dbp_predictions, cal_dbps)

    sbp_mse += tst_sbp_mse
    dbp_mse += tst_dbp_mse
    for i in range(num_of_cal_samples):
        output_file.write(fnames[i] + "," + str(int(cal_sbps[i]))
                + "," + str(int(cal_dbps[i])) + "\n")

    for i in range(num_of_cal_samples,len(data)):
        output_file.write(fnames[i] + "," + str(int(tst_sbp_predictions[i-num_of_cal_samples]))
                + "," + str(int(tst_dbp_predictions[i-num_of_cal_samples])) + "\n")

print(sbp_mse)
print(dbp_mse)
print(sbp_mse + 2 * dbp_mse)


