from tensorflow import keras
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import InputSpec
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import activations,regularizers, initializers,constraints
#from tensorflow.keras.utils import conv_utils
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects


def add_channels(input_tensor):

    """
    input_tensor: (batch, 1, 1, c), or (batch, x_dim, y_dim, c)
    In the first case, first tile the input_tensor to be (batch, x_dim, y_dim, c)
    In the second case, skiptile, just concat
    """

    #tf.dtypes.cast(x, tf.int32)
    x_dim = tf.dtypes.cast(tf.shape(input_tensor)[1],  tf.int32 )#m
    y_dim = tf.dtypes.cast(tf.shape(input_tensor)[2],  tf.int32 )#m

    """    
    if not self.skiptile:
        input_tensor = tf.tile(input_tensor, [1, self.x_dim, self.y_dim, 1]) # (batch, 64, 64, 2)
        input_tensor = tf.cast(input_tensor, 'float32')
    """

    batch_size_tensor = tf.shape(input_tensor)[0]  # get batch size

    xx_ones = tf.ones([batch_size_tensor, x_dim],
                        dtype=tf.int32)                       # e.g. (batch, 64)
    xx_ones = tf.expand_dims(xx_ones, -1)                   # e.g. (batch, 64, 1)
    xx_range = tf.tile(tf.expand_dims(tf.range(y_dim), 0),
                        [batch_size_tensor, 1])             # e.g. (batch, 64)
    xx_range = tf.expand_dims(xx_range, 1)                  # e.g. (batch, 1, 64)


    xx_channel = tf.matmul(xx_ones, xx_range)               # e.g. (batch, 64, 64)
    xx_channel = tf.expand_dims(xx_channel, -1)             # e.g. (batch, 64, 64, 1)


    yy_ones = tf.ones([batch_size_tensor, y_dim],
                        dtype=tf.int32)                       # e.g. (batch, 64)
    yy_ones = tf.expand_dims(yy_ones, 1)                    # e.g. (batch, 1, 64)
    yy_range = tf.tile(tf.expand_dims(tf.range(x_dim), 0),
                        [batch_size_tensor, 1])             # (batch, 64)
    yy_range = tf.expand_dims(yy_range, -1)                 # e.g. (batch, 64, 1)

    yy_channel = tf.matmul(yy_range, yy_ones)               # e.g. (batch, 64, 64)
    yy_channel = tf.expand_dims(yy_channel, -1)             # e.g. (batch, 64, 64, 1)

    xx_channel = tf.cast(xx_channel, 'float32') / (tf.cast(x_dim, 'float32') - 1)
    yy_channel = tf.cast(yy_channel, 'float32') / (tf.cast(y_dim, 'float32') - 1)
    xx_channel = xx_channel*2 - 1                           # [-1,1]
    yy_channel = yy_channel*2 - 1

    ret = tf.concat([input_tensor,
                        xx_channel,
                        yy_channel], axis=-1)    # e.g. (batch, 64, 64, c+2)

    return ret

if __name__=='__main__':
    img_ip = Input(shape=(224,224,3))

    
    #cc_layer = CoordConv2D(filters=16,kernel_size=3,activation='relu')(img_ip)