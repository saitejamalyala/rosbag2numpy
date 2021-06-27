
from numpy import float16
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow as tf

list_mask=[[1., 1.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 0.],
       [1., 1.]]


class custom_mask(layers.Layer):
    """Layer that masks tensor at specific locations as mentioned in binary tensor 

    Args:
        layers ([type]): [description]
    """    
    def __init__(self, list_mask,trainable=False, name="mask_at_position",**kwargs):

        super().__init__(trainable=trainable, name=name,**kwargs)
        self.list_mask = list_mask

    def call(self, inputs, **kwargs):
        temp = inputs
        mask = tf.constant(self.list_mask,dtype=tf.float32)
        # masking with first and last co-ordinate
        first_last_skip_conn = tf.math.multiply(mask, temp)
        output = first_last_skip_conn
        return output



def nn(full_skip:bool=True):

    # Grid Map input
    ip_gridmap = layers.Input(shape=(1536,1536,1))

    #CNN - branch1
    #1x1 conv 
    #x_A = layers.Conv2D(3,kernel_size=1,strides=1)(ip_gridmap)
    
    x_A = layers.Conv2D(16,kernel_size=7,strides=2)(ip_gridmap)
    x_A = layers.ReLU()(x_A)
    x_A = layers.BatchNormalization()(x_A)
    x_A = layers.AvgPool2D(pool_size=(4,4))(x_A)

    x_A = layers.Conv2D(32,kernel_size=5,strides=2)(x_A)
    x_A = layers.ReLU()(x_A)
    x_A = layers.BatchNormalization()(x_A)
    x_A = layers.AvgPool2D(pool_size=(4,4))(x_A)

    
    x_A = layers.Conv2D(64,kernel_size=3,strides=2)(x_A)
    x_A = layers.BatchNormalization()(x_A)
    x_A = layers.ReLU()(x_A)
    x_A = layers.AvgPool2D(pool_size=(2,2))(x_A)


    x_A = layers.Flatten()(x_A)


    # Other inputs
    ip_grid_org_res = layers.Input(shape=(3,),name="Grid_origin_res")
    ip_left_bnd = layers.Input(shape=(25,2),name="Left_boundary")
    ip_right_bnd = layers.Input(shape=(25,2),name="Right_boundary")
    ip_car_odo = layers.Input(shape=(3,),name="Car_loc")
    ip_init_path = layers.Input(shape=(25,2),name="Initial_path")
    #ip_filedetais = layers.Input

    # branch 5
    conc_grid_orgres_car_odo = layers.concatenate([ip_grid_org_res,ip_car_odo])

    #reshaping paths
    reshape_init_path = layers.Reshape((50,))(ip_init_path)
    reshape_left_bnd = layers.Reshape((50,))(ip_left_bnd)
    reshape_right_bnd = layers.Reshape((50,))(ip_right_bnd)

    
    #concatenate feature
    concat_feat = layers.concatenate([x_A, reshape_init_path, reshape_left_bnd, reshape_right_bnd, conc_grid_orgres_car_odo])


    # Dense Network
    output = layers.Dense(128, activation='linear')(concat_feat)
    output = layers.BatchNormalization()(output)
    output = layers.ReLU()(output)
    
    
    output = layers.Dense(96, activation='linear')(output)
    output = layers.BatchNormalization()(output)
    output = layers.ReLU()(output)
    
    
    output = layers.Dense(64, activation='linear')(output)
    output = layers.BatchNormalization()(output)
    output = layers.ReLU()(output)
    #output = layers.LeakyReLU()(output)

    
    output = layers.Dense(50, activation='linear')(output)

    if full_skip:
        output = layers.add([output,reshape_init_path])
        output = layers.Dense(50, activation='linear')(output)
    else:
        """
        first_last_skip_conn = tf.constant(list_mask,dtype=tf.float32)
        # masking with first and last co-ordinate
        first_last_skip_conn = tf.math.multiply(first_last_skip_conn,ip_init_path)
        """
        first_last_skip_conn= custom_mask(list_mask=list_mask)(ip_init_path)
        reshape_first_last_skip = layers.Reshape((50,))(first_last_skip_conn)
        output = layers.add([output, reshape_first_last_skip])
        output = layers.Dense(50, activation='linear')(output)

    output = layers.Reshape((25,2))(output)
    
    nn_fun = models.Model(inputs = [ip_gridmap,ip_grid_org_res,ip_left_bnd, ip_right_bnd, ip_car_odo, ip_init_path], outputs= output)

    nn_fun.summary(line_length=120)
    
    return nn_fun

#nn(full_skip=False)


