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
