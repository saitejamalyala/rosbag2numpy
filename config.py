#from tensorflow.keras.metrics import MeanAbsoluteError
from rosbag2numpy.losses import *

params = {
    "Description": "New data(only parking), coord conv, Normalized co-ordinates (paths, boundaries, ego/car odo, values range from 0-1536(grid min-grid max))",

    #-------------- Data set Parameters ------------#
    "H_BATCH_SIZE" :32,
    "H_SHUFFLE_BUFFER" : 32*100,
    "normalize_coords": True,
    "normalize_factor": 1536,
    #"dataset_dir":'/netpool/work/gpu-3/users/malyalasa/New_folder/tf_records',
    "dataset_dir":'/bigpool/projects/yao_SCANGAN360/New_Folder/tf_records',
    #------------ Model Hyper paramaters ----------#
    "epochs":50,
    "lr":0.02,
    "optimizer":'adam',
    "metric":'accuracy',#MeanAbsoluteError(),
    "losses":[euclidean_distance_loss,endpoint_loss],
    "loss_weights":[1,2],
    #----------- directory paths -----------------#
    "log_dir": '/netpool/work/gpu-3/users/malyalasa/New_folder/rosbag2numpy/logging',


    #---------- Model params -----------#
    "full_skip":False,
    "drop_rate":{
        "dense_rate1":0.4,
        "dense_rate2":0.8,
        "dense_rate3":0.0,
    }
}


generalization_model_params = {
    "full_skip":None,
    "drop_rate":{
        "dense_rate1":0.5,
        "dense_rate2":0.1,
        "dense_rate3":0.1,
    }

}

