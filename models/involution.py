import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
from tensorflow.keras import models


class Involution2D(layers.Layer):

    def __init__(self, filters, kernel_size = 3, strides = 1, padding = 'same', dilation_rate = 1, groups = 1, reduce_ratio = 1):
        super(Involution2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.groups = groups
        self.reduce_ratio = reduce_ratio
        self.reduce_mapping = tf.keras.Sequential(
            [
                layers.Conv2D(filters // reduce_ratio, 1, padding = padding), 
                layers.BatchNormalization(), 
                layers.Activation('relu'), 
            ]
        )
        self.span_mapping = layers.Conv2D(kernel_size * kernel_size * groups, 1, padding = padding)
        self.initial_mapping = layers.Conv2D(self.filters, 1, padding = padding)
        if strides > 1:
            self.o_mapping = layers.AveragePooling2D(strides)
    
    def call(self, x):
        weight = self.span_mapping(self.reduce_mapping(x if self.strides == 1 else self.o_mapping(x)))
        _, h, w, c = K.int_shape(weight)
        weight = K.expand_dims(K.reshape(weight, (-1, h, w, self.groups, self.kernel_size * self.kernel_size)), axis = 4)
        out = tf.image.extract_patches(images = x if self.filters == c else self.initial_mapping(x),  
                                       sizes = [1, self.kernel_size, self.kernel_size, 1], 
                                       strides = [1, self.strides, self.strides, 1], 
                                       rates = [1, self.dilation_rate, self.dilation_rate, 1], 
                                       padding = "SAME" if self.padding == 'same' else "VALID")
        out = K.reshape(out, (-1, h, w, self.groups, self.filters // self.groups, self.kernel_size * self.kernel_size))
        out = K.sum(weight * out, axis = -1)
        out = K.reshape(out, (-1, h, w, self.filters))
        return out

def nn():

    ip_gridmap = layers.Input(shape=(1536,1536,1))
    x1 = layers.Flatten()(ip_gridmap)
    x1 = layers.Dense(8,activation='relu')(x1)

    # Other inputs
    ip_grid_org_res = layers.Input(shape=(3,),name="Grid_origin_res")
    ip_left_bnd = layers.Input(shape=(25,2),name="Left_boundary")
    ip_right_bnd = layers.Input(shape=(25,2),name="Right_boundary")
    ip_car_odo = layers.Input(shape=(3,),name="Car_loc")
    ip_init_path = layers.Input(shape=(25,2),name="Initial_path")

    #reshaping paths
    reshape_init_path = layers.Reshape((50,))(ip_init_path)
    reshape_left_bnd = layers.Reshape((50,))(ip_left_bnd)
    reshape_right_bnd = layers.Reshape((50,))(ip_right_bnd)

    concat_all = layers.concatenate([x1,reshape_left_bnd, reshape_init_path, reshape_right_bnd, ip_grid_org_res, ip_car_odo])

    output = layers.Dense(64,activation='relu')(concat_all)
    output = layers.Dense(128,activation='relu')(output)

    output = layers.Dense(50,activation='sigmoid')(output)

    output = layers.Reshape((25,2))(output)
    
    nn_fun = models.Model(inputs = [ip_gridmap,ip_grid_org_res,ip_left_bnd, ip_right_bnd, ip_car_odo, ip_init_path], outputs= output)

    nn_fun.summary(line_length=120)

    return nn_fun

model=nn()
print(model.summary())

if '__name__'=='__main__':
   model=nn()
   print(model.summary())