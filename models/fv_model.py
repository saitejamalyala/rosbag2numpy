from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow as tf
import wandb
from rosbag2numpy import SEED
tf.random.set_seed(SEED)
import sys
sys.path.append("../")
sys.path.append('./')
sys.path.append('.././')


def TimeDistributedDense(params):

    ip_fv = layers.Input(shape=(25,1040))
    x_A = layers.TimeDistributed(layers.Dense(units=params.get("layer1_Dense_units")))(ip_fv)
    #x_A = layers.TimeDistributed(layers.Dense(units=2,activation='tanh',activity_regularizer = tf.keras.regularizers.l2()))(x_A)
    x_A = layers.TimeDistributed(layers.Dense(units=2,activation='tanh'))(x_A)

    nn_fun = models.Model(inputs = ip_fv, outputs= x_A)

    nn_fun.summary()

    return nn_fun

def BidiLSTM():
    ip_fv = layers.Input(shape=(25,1040))

    x_A = layers.Bidirectional(layers.LSTM(units=1,return_sequences=True))(ip_fv)

    x_A = layers.Bidirectional(layers.LSTM(units=25))(x_A)

    x_A = layers.Reshape(target_shape=(25,2))(x_A)

    nn_fun = models.Model(inputs = ip_fv, outputs= x_A)

    nn_fun.summary()

    return nn_fun

def LSTMmodel(params):
    ip_fv = layers.Input(shape=(25,1040))

    x_A = layers.LSTM(units=params.get("layer1_LSTM_units"),return_sequences=True)(ip_fv)

    x_A = layers.LSTM(units=50)(x_A)

    x_A = layers.Reshape(target_shape=(25,2))(x_A)

    nn_fun = models.Model(inputs = ip_fv, outputs= x_A)

    nn_fun.summary()

    return nn_fun

def hybridmodel(params):
    ip_fv = layers.Input(shape=(25,1040))

    x_A = layers.LSTM(units=params.get("layer1_LSTM_units"),return_sequences=True)(ip_fv)

    x_A = layers.TimeDistributed(layers.Dense(units=2))(x_A)

    #x_A = layers.Reshape(target_shape=(25,2))(x_A)

    nn_fun = models.Model(inputs = ip_fv, outputs= x_A)

    nn_fun.summary()

    return nn_fun


#BidiLSTM()