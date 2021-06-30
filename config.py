#from tensorflow.keras.metrics import MeanAbsoluteError
from losses import *

params = {
    "Description": "reproducing scenario wise split , string not updated in previous run",

    #-------------- Data set Parameters ------------#
    "H_BATCH_SIZE" :32,
    "H_SHUFFLE_BUFFER" : 32*100,

    "dataset_dir":'/netpool/work/gpu-3/users/malyalasa/New_folder/tf_records',
    #------------ Model Hyper paramaters ----------#
    "epochs":50,
    "lr":0.02,
    "optimizer":'adam',
    "metric":'accuracy',#MeanAbsoluteError(),
    "losses":[euclidean_distance_loss,endpoint_loss],
    "loss_weights":[1,0],
    #----------- directory paths -----------------#
    "log_dir": '/netpool/work/gpu-3/users/malyalasa/New_folder/rosbag2numpy/logging',


    #---------- Model params -----------#
    "full_skip":False,
    "drop_rate":{
        "dense_rate1":0.0,
        "dense_rate2":0.0,
        "dense_rate3":0.0,
    }
}


generalization_model_params = {
    "full_skip":True,
    "drop_rate":{
        "dense_rate1":0.5,
        "dense_rate2":0.1,
        "dense_rate3":0.1,
    }

}

