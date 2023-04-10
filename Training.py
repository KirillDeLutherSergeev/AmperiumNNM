import numpy as np
import math
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
import tensorflow.keras.backend as K

mae = tf.keras.losses.MeanAbsoluteError()
mse = tf.keras.losses.MeanSquaredError()

def partition_data(input, output, sequenceLength, trainTestRatio=0.8, overlap=256):
    numSamples = np.minimum(input.shape[0], output.shape[0])
    numBatches = math.floor((numSamples - sequenceLength) / overlap)

    batchedInput = []
    batchedOutput = []
    
    for i in range(0, numBatches):
        start = i * overlap
        end = start + sequenceLength
        batchedInput.append(np.array(input[start:end]))
        batchedOutput.append(np.array(output[start:end]))

    trainLength = math.floor(numBatches * trainTestRatio)
    
    trainInput = np.array(batchedInput[:trainLength])
    trainOutput = np.array(batchedOutput[:trainLength])
    testInput = np.array(batchedInput[trainLength:])
    testOutput = np.array(batchedOutput[trainLength:])

    return (trainInput, trainOutput, testInput, testOutput)

def HiPass(x, coeff=0.85):
    return tf.concat([x[:, 0:1, :], x[:, 1:, :] - coeff * x[:, :-1, :]], axis=1)

def LowPass(x, coeff=0.85):
    return tf.concat([x[:, 0:1, :], x[:, 1:, :] + coeff * x[:, :-1, :]], axis=1)

def esr(target_y, predicted_y): 
    return K.sum(tf.pow(target_y - predicted_y, 2), axis=0) / (K.sum(tf.pow(target_y, 2), axis=0) + 1e-10)

def mae_emphasized(target_y, predicted_y):
    return mae(HiPass(target_y), HiPass(predicted_y))

def mse_emphasized(target_y, predicted_y):
    return mse(HiPass(target_y), HiPass(predicted_y))

def esr_emphasized(target_y, predicted_y):
    tgtY = HiPass(target_y)
    predY = HiPass(predicted_y)
    return K.sum(tf.pow(tgtY - predY, 2), axis=0) / (K.sum(tf.pow(tgtY, 2), axis=0) + 1e-10)

def build_model(useD1=True, useC1=True, useC2=True, loss='mse', metrics=[esr], learningRate=0.008, epsilon=1.e-08, hiddenSize=16, conv1Size=128, conv2Size=2048, showInfo=False):
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

    if useC1:
        conv_init = np.zeros(conv1Size)
        conv_init[0] = 1.0
        model.add(Conv1D(filters=1, kernel_size=conv1Size,
            padding='causal',
            bias_initializer='zeros',
            kernel_initializer=tf.constant_initializer(conv_init),
            use_bias=False,
            name='C1'))

    model.add(LSTM(name='L', units=hiddenSize,
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

    model.compile(optimizer=Adam(learning_rate=learningRate, epsilon=epsilon), loss=loss, metrics=metrics)

    if showInfo:
        model.summary()

    return(model)

def prepare_dataset(x_all, y_all, numSteps = 2048, trainSize=0.8, overlap=256, batchSize=64):
    x_train, y_train, x_test, y_test = partition_data(x_all, y_all, numSteps, trainSize, overlap)

    trainObservations = int(x_train.shape[0] / batchSize) * batchSize
    testObservations = int(x_test.shape[0] / batchSize) * batchSize

    x_train = x_train[:trainObservations]
    y_train = y_train[:trainObservations]
    x_test = x_test[:testObservations]
    y_test = y_test[:testObservations]

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)    
    
    return(x_train, y_train, x_test, y_test)

def scheduler(epoch, lr):
    if 0 < epoch < 12:
        return lr * 0.95
    return lr

class ModelSaverCallback(tf.keras.callbacks.Callback):

    def __init__(self, initial_model, epochs):
        self.best_val_loss = 1000
        self.current_epoch = 0
        self.num_epochs = epochs
        self.batch = 0
        self.count = 0
        self.best_model = initial_model

    def return_best_model(self):
        return self.best_model

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        if logs['val_loss'] < self.best_val_loss:
            self.best_val_loss = logs['val_loss']
            self.best_model.set_weights(self.model.get_weights())

def train_model(x_train, y_train, x_test, y_test, model, epochs=8, batchSize=64, shuffle=True):
    scheduler_clbk = tf.keras.callbacks.LearningRateScheduler(scheduler)
    plateu_clbk = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    earlystop_clbk = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=5, verbose=1)
    modelSaver_clbk = ModelSaverCallback(initial_model=model, epochs=epochs)
    
    history = model.fit(x=x_train,y=y_train, validation_data=(x_test, y_test),
        epochs=epochs, 
        verbose=1,
        batch_size=batchSize,
        callbacks=[scheduler_clbk, plateu_clbk, earlystop_clbk, modelSaver_clbk], shuffle=shuffle)

    model = modelSaver_clbk.return_best_model()