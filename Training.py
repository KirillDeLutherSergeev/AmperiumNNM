from AudioUtils import *
import numpy as np

def partition_data(input_data, output_data, sequence_length, trainTestRatio=0.7, overlap=2048, normalize=True):
    if normalize:
        input = normalize_max_peak(input_data)
        output = normalize_at_minus_6dB(output_data)
    else:
        input = input_data
        output = output_data

    numSamples = np.minimum(input.shape[0], output.shape[0])
    numBatches = math.floor((numSamples - sequence_length) / overlap)

    batchedInput = []
    batchedOutput = []
    
    for i in range(0, numBatches):
        start = i * overlap
        end = start + sequence_length
        batchedInput.append(np.array(input[start:end]))
        batchedOutput.append(np.array(output[start:end]))

    trainLength = math.floor(numBatches * trainTestRatio)
    
    trainInput = np.array(batchedInput[:trainLength])
    trainOutput = np.array(batchedOutput[:trainLength])
    testInput = np.array(batchedInput[trainLength:])
    testOutput = np.array(batchedOutput[trainLength:])

    return (trainInput, trainOutput, testInput, testOutput)

def build_model(useD1=True, useC1=True, useC2=True, loss='mae', conv1Size=128, conv2Size=2048):
    # Create Sequential Model ###########################################
    clear_session()

    model = Sequential()

    model.add(InputLayer(input_shape=(None, 1)))

    if useD1:
        model.add(Dense(units=1, use_bias=False,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=None),
            bias_initializer='zeros',
            activation='tanh',
            name='D1'))

    if useC2:
        conv_init = np.zeros(conv1Size)
        conv_init[0] = 1.0
        model.add(Conv1D(filters=1, kernel_size=conv1Size,
            padding='causal',
            bias_initializer='zeros',
            kernel_initializer=tf.constant_initializer(conv_init),
            use_bias=False,
            name='C1'))

    model.add(LSTM(name='L', units=hidden_units,
        stateful=False,
        bias_initializer='zeros',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=None),
        recurrent_initializer=tf.keras.initializers.Orthogonal(seed=None),
        return_sequences=True))

    model.add(Dense(units=1,
        bias_initializer='zeros',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=None),
        name='D2'))

    if useC2:
        conv_init = np.zeros(conv2Size)
        conv_init[0] = 1.0
        model.add(Conv1D(filters=1, kernel_size=conv2Size,
            padding='causal',
            bias_initializer='zeros',
            kernel_initializer=tf.constant_initializer(conv_init),
            use_bias=False,
            name='C2'))

    model.compile(optimizer=Adam(learning_rate=learning_rate, epsilon=epsilon), loss=loss, metrics='mse')

    model.summary()

    return(model)
