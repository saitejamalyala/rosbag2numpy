from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow as tf
from ..config import params

def TimeDistributedDense():
    ip_fv = layers.Input(shape=(25,1040))
    x_A = layers.TimeDistributed(layers.Dense(units=params.get("layer_1_units")))(ip_fv)
    x_A = layers.TimeDistributed(layers.Dense(units=2,activation='tanh'))(x_A)

    nn_fun = models.Model(inputs = ip_fv, outputs= x_A)

    nn_fun.summary()

    return nn_fun


def BidiLSTM():
    ip_fv = layers.Input(shape=(25,1040))

    x_A = layers.Bidirectional(layers.LSTM(units=5,return_sequences=True))(ip_fv)

    x_A = layers.Bidirectional(layers.LSTM(units=25,activity_regularizer=tf.keras.regularizers.l1(0.1)))(x_A)

    x_A = layers.Reshape(target_shape=(25,2))(x_A)

    nn_fun = models.Model(inputs = ip_fv, outputs= x_A)

    nn_fun.summary()

    return nn_fun

#BidiLSTM()
