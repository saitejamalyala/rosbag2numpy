import sys
sys.path.append("/netpool/work/gpu-3/users/malyalasa/New_folder")
from tensorflow.keras import layers
from tensorflow.keras import models
from rosbag2numpy.config import params
import os
#print(tf.__version__)
os.environ['CUDA_VISIBLE_DEVICES']="3"

def encoder_nw(params):
    """Encoder to convert grid map to one dimensional encoded vector
    """
    # Grid Map input
    ip_gridmap = layers.Input(shape=(1536,1536,1))

    # Block 1
    
    x_A = layers.Conv2D(16,kernel_size=5,strides=2)(ip_gridmap)
    x_A = layers.LeakyReLU()(x_A)
    x_A = layers.BatchNormalization()(x_A)
    x_A = layers.AvgPool2D(pool_size=(4,4))(x_A)
    
    
    # Block 2
    
    x_A = layers.Conv2D(32,kernel_size=3,strides=2)(x_A)
    x_A = layers.LeakyReLU()(x_A)
    x_A = layers.BatchNormalization()(x_A)
    x_A = layers.AvgPool2D(pool_size=(2,2))(x_A)
    

    x_A = layers.Conv2D(32,kernel_size=3,strides=2)(x_A)
    x_A = layers.LeakyReLU()(x_A)
    x_A = layers.BatchNormalization()(x_A)
    #x_A = layers.AvgPool2D(pool_size=(2,2))(x_A)

    x_A = layers.Conv2D(8,kernel_size=1,strides=2)(x_A)

    x_A = layers.Conv2D(2,kernel_size=1,strides=2)(x_A)

    x_A = layers.Flatten()(x_A)

    output = layers.Dense(50)(x_A)

    nn_fun = models.Model(inputs = ip_gridmap, outputs= output)

    nn_fun.summary(line_length=120)

    return nn_fun

#model = encoder_nw(params)
#model.save('/netpool/work/gpu-3/users/malyalasa/New_folder/rosbag2numpy/models/encoder_nw.h5',save_format ='h5')

if '__name__'=='__main__':
   model=encoder_nw(params)
   model.summary()
