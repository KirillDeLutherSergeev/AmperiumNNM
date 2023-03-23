mae = tf.keras.losses.MeanAbsoluteError()
mse = tf.keras.losses.MeanSquaredError()

def HiPass(x, coeff=0.85):
    return tf.concat([x[:, 0:1, :], x[:, 1:, :] - coeff * x[:, :-1, :]], axis=1)

def LowPass(x, coeff=0.85):
    return tf.concat([x[:, 0:1, :], x[:, 1:, :] + coeff * x[:, :-1, :]], axis=1)

def esr_loss_with_frequency(target_y, predicted_y):
    return mae(HiPass(target_y), HiPass(predicted_y))

def esr_loss(target_y, predicted_y):
    return mae(target_y, predicted_y)

def error_to_signal(y_true, y_pred): 
    """
    Error to signal ratio with pre-emphasis filter:
    """
    y_true, y_pred = HiPass(y_true), HiPass(y_pred)
    return K.sum(tf.pow(y_true - y_pred, 2), axis=0) / (K.sum(tf.pow(y_true, 2), axis=0) + 1e-10)
