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
