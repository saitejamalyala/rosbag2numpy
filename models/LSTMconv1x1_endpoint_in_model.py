
import sys
sys.path.append("/netpool/work/gpu-3/users/malyalasa/New_folder")
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow as tf
from tensorflow.keras.regularizers import l1,l1_l2,l2
from tensorflow.python.keras.regularizers import L1
#from rosbag2numpy.config import params
from rosbag2numpy import config as params
import os
#print(tf.__version__)
#from ..config import params
os.environ['CUDA_VISIBLE_DEVICES']="3"

#@tf.keras.utils.register_keras_serializable()
class CustomMaskLayer(layers.Layer):
    """Layer that masks tensor at specific locations as mentioned in binary tensor 

    Args:
        layers (layers.Layer): keras.layers baseclass
    """    

    def __init__(self, list_mask=[[1., 1.],
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
                                [1., 1.]],
                                name=None,**kwargs):

        self.list_mask = list_mask
        super(CustomMaskLayer,self).__init__(name=name,**kwargs)
        

    def call(self, inputs):
        temp = inputs
        mask = tf.constant(self.list_mask,dtype=tf.float32)
        # masking with first and last co-ordinate
        first_last_skip_conn = tf.math.multiply(mask, temp)
        output = first_last_skip_conn
        return output

    def get_config(self):

        config = super(CustomMaskLayer,self).get_config()
        config.update({
            "list_mask": self.list_mask,
        })
        return config

def nn(full_skip=False,params=None):
    fp_list_mask=[[1., 1.],
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
                [0., 0.]]
    # Grid Map input
    ip_gridmap = layers.Input(shape=(1536,1536,1))

    #CNN - branch1
    #1x1 conv 
    #x_A = layers.Conv2D(3,kernel_size=1,strides=1)(ip_gridmap)
    # Block 1
    
    x_A = layers.Conv2D(16,kernel_size=3,strides=2)(ip_gridmap)
    x_A = layers.LeakyReLU()(x_A)
    x_A = layers.BatchNormalization()(x_A)
    x_A = layers.AvgPool2D(pool_size=(4,4))(x_A)
      
    # Block 2
    
    x_A = layers.Conv2D(32,kernel_size=3,strides=1)(x_A)
    x_A = layers.LeakyReLU()(x_A)
    x_A = layers.BatchNormalization()(x_A)
    x_A = layers.AvgPool2D(pool_size=(4,4))(x_A)
    
    # 1x1 blocks
    """
        x_A = layers.Conv2D(16,kernel_size=1,strides=1)(x_A)
        x_A = layers.ReLU()(x_A)
        x_A = layers.BatchNormalization()(x_A)
    """
    x_A = layers.Conv2D(8,kernel_size=1,strides=2)(x_A)

    x_A = layers.Conv2D(2,kernel_size=1,strides=1)(x_A)


    # Block 3
    """
        x_A = layers.Conv2D(64,kernel_size=3,strides=2)(x_A)
        x_A = layers.BatchNormalization()(x_A)
        x_A = layers.ReLU()(x_A)
        x_A = layers.AvgPool2D(pool_size=(2,2))(x_A)
    """
    x_A = layers.Flatten()(x_A)


    # Other inputs
    ip_grid_org_res = layers.Input(shape=(3,),name="Grid_origin_res")
    ip_left_bnd = layers.Input(shape=(25,2),name="Left_boundary")
    ip_right_bnd = layers.Input(shape=(25,2),name="Right_boundary")
    ip_car_odo = layers.Input(shape=(3,),name="Car_loc")
    ip_init_path = layers.Input(shape=(25,2),name="Initial_path")
    ip_file_name = layers.Input(shape=(1,),name="File_name",dtype=tf.string)

    #ip_filedetais = layers.Input

    # branch 5
    conc_grid_orgres_car_odo = layers.concatenate([ip_grid_org_res,ip_car_odo])

    #reshaping paths
    reshape_init_path = layers.Reshape((50,))(ip_init_path)
    reshape_left_bnd = layers.Reshape((50,))(ip_left_bnd)
    reshape_right_bnd = layers.Reshape((50,))(ip_right_bnd)
    
    #concatenate feature
    concat_feat = layers.concatenate([x_A, reshape_left_bnd, reshape_right_bnd, conc_grid_orgres_car_odo,reshape_init_path ])
    #tf.print(type(concat_feat.shape))
    #tf.print((concat_feat.shape.concatenate(1).as_list()[1:]))
    
    #output = layers.Reshape(target_shape=(concat_feat.shape.concatenate(1).as_list()[1:]))(concat_feat)
    #output = layers.LSTM(units=25,return_sequences=True)(output)
    #output = layers.LSTM(units=50)(output)
    
    
    output = layers.Dense(50,activation='relu')(concat_feat)

    if full_skip:
        # Block 6-fs
        output = layers.add([output,reshape_init_path])
        output = layers.Reshape(target_shape=(output.shape.concatenate(1).as_list()[1:]))(output)
        output = layers.LSTM(units=25,return_sequences=True)(output)
        output = layers.LSTM(units=50)(output)

        output = layers.Dense(50, activation=params.get("lastlayer_activation"))(output)

    else:
        """
        #Implementation without CustomMaskLayer
        first_last_skip_conn = tf.constant(list_mask,dtype=tf.float32)
        # masking with first and last co-ordinate
        first_last_skip_conn = tf.math.multiply(first_last_skip_conn,ip_init_path)
        """
        # Block 6-endpoints_condition
        if full_skip==False:
            first_last_skip_conn= CustomMaskLayer()(ip_init_path)
            reshape_first_last_skip = layers.Reshape((50,))(first_last_skip_conn)
            output = layers.add([output, reshape_first_last_skip])
            output = layers.Dense(50, activation=params.get("lastlayer_activation"))(output)

        # only first point skip connection(use full_skip=none)  
        else:   
            first_last_skip_conn = tf.constant(fp_list_mask,dtype=tf.float32)
            # masking with first 
            first_last_skip_conn = tf.math.multiply(first_last_skip_conn,ip_init_path)
            reshape_first_last_skip = layers.Reshape((50,))(first_last_skip_conn)
            output = layers.add([output, reshape_first_last_skip])
            output = layers.Dense(50, activation=params.get("lastlayer_activation"))(output)
    

    #output
    output = layers.Reshape((25,2))(output)
    
    nn_fun = models.Model(inputs = [ip_gridmap,ip_grid_org_res,ip_left_bnd, ip_right_bnd, ip_car_odo, ip_init_path, ip_file_name], outputs= output)

    nn_fun.summary(line_length=120)
    
    return nn_fun

#nn(full_skip=True,params=params)

if '__name__'=='__main__':
   model=nn(full_skip=False)
   model.summary()



