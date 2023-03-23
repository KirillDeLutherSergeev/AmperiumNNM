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

def partition_data(input_data, output_data, sequence_length, trainTestRatio=0.7, overlap=2048, normalize=True):
    if normalize:
        input = normalize_max_peak(input_data)
        output = normalize_at_minus_3dB(output_data)
    else:
        input = input_data
        output = output_data

    numSamples = np.minimum(input.shape[0], output.shape[0])
    
    reshapedInput = []
    reshapedOutput = []
    num_batches = math.floor((numSamples - sequence_length) / overlap)
    
    for i in range(0, num_batches):
        start = i * overlap
        end = start + sequence_length
        reshapedInput.append(np.array(input[start:end]))
        reshapedOutput.append(np.array(output[start:end]))

    train_length = math.floor(num_batches * trainTestRatio)
    
    input_train = np.array(reshapedInput[:train_length])
    output_train = np.array(reshapedOutput[:train_length])
    input_test = np.array(reshapedInput[train_length:])
    output_test = np.array(reshapedOutput[train_length:])

    return (input_train, output_train, input_test, output_test)
