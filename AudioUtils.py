import os
from scipy.io import wavfile
import numpy as np
import math

def save_wav(name, data, fs = 44100):
    wavfile.write(name, fs, data.flatten().astype(np.float32))

def linear_2_db(signal):
    return 20.0 * np.log10(signal)

def db_2_linear(signal_db):
    return 10.0 ** (signal_db / 20.0)

def normalize_max_peak(signal):
    data_max = np.max(np.abs(signal))
    return signal / data_max

def normalize_at_minus_3dB(signal):
    data_max = np.max(np.abs(signal))
    minus_3db_in_linear = 0.707
    return signal * (minus_3db_in_linear / data_max)

def normalize_at_minus_6dB(signal):
    data_max = np.max(np.abs(signal))
    return signal * (0.5 / data_max)

def calculate_rms(signal):
    rms = np.sqrt(np.mean(signal ** 2))
    return rms

def calculate_rms_db(signal):
    rms = calculate_rms(signal)
    rms_db = linear_2_db(rms)
    return rms_db

def calculate_peak_db(signal):
    data_max = np.max(np.abs(signal))
    peak_db = linear_2_db(data_max)
    return peak_db

def load_audio_data(inFile, outFile, offsetSec=1, delay=0):
    # Load and Preprocess Data ###########################################
    in_rate, in_data = wavfile.read(inFile)
    out_rate, out_data = wavfile.read(outFile)

    assert in_rate == out_rate, "Mismatched sample rates"

    offset = int(math.floor(offsetSec * in_rate))

    x_all = in_data.astype(np.float32).flatten()
    x_all = x_all[offset:]  

    y_all = out_data.astype(np.float32).flatten()

    offsetOut = offset-delay
    if offsetOut < 0:
        offsetOut = 0
        
    y_all = y_all[offsetOut:]

    x_all = normalize_max_peak(x_all).reshape(len(x_all),1)
    y_all = normalize_at_minus_3dB(y_all).reshape(len(y_all),1)

    return(x_all, y_all, in_rate)

def check_if_model_exists(name, modelPath='models/'):
    if not os.path.exists(modelPath+name):
        os.makedirs(modelPath+name)
    else:
        print("A model with the same name already exists. Please choose a new name.")
        exit
