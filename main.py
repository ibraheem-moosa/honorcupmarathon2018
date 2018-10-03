import sys
import os
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from scipy.fftpack import fft
import numpy as np
import random

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
        ecg = [int(val[0]) for val in data[1:]]
        ppg = [int(val[1]) for val in data[1:]]
        dataset[user_id].append((sbp, dbp, ecg, ppg))


ecg_num_freqs = 10
ppg_num_freqs = 10

reg = MLPRegressor(verbose=True)
reg = LinearRegression(n_jobs=-1)
multi_reg = MultiOutputRegressor(reg, n_jobs=-1)

for user in dataset:
    data = dataset[user]
    num_of_samples = len(data)
    num_of_cal_samples = num_of_samples // 5
    print(user, num_of_samples)
    indices = list(range(len(data))) 
    random.shuffle(indices)

    features = []
    outputs = []
    for sample in data:
        sbp, dbp, ecg, ppg = sample
        outputs.append(np.array([sbp, dbp]))
        ecg_fft = np.absolute(fft(ecg))[1:ecg_num_freqs]
        ppg_fft = np.absolute(fft(ppg))[1:ppg_num_freqs]
        features.append(np.hstack((ecg_fft, ppg_fft)))

    cal_features = np.vstack([features[i] for i in indices[:num_of_cal_samples]])
    cal_outputs = np.vstack([outputs[i] for i in indices[:num_of_cal_samples]])

    tst_features = np.vstack([features[i] for i in indices[num_of_cal_samples:]])
    tst_outputs = np.vstack([outputs[i] for i in indices[num_of_cal_samples:]])
    
    multi_reg.fit(cal_features, cal_outputs)
    print(multi_reg.score(tst_features, tst_outputs))

