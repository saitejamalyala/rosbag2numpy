#from tensorflow.keras.metrics import MeanAbsoluteError
from rosbag2numpy.losses import *

params = {
    #"Description": "Costmaps (distance and direction),New data(only parking), coord conv, Normalized co-ordinates (paths, boundaries, ego/car odo, values range from 0-1536(grid min-grid max))",
    "Description": "Testing new loss",
    #-------------- Data set Parameters ------------#
    "H_BATCH_SIZE" :32,
    "H_SHUFFLE_BUFFER" : 32*100,
    "normalize_coords": True,
    "normalize_factor": 1536.0,
    "lastlayer_activation":'sigmoid',
    #"dataset_dir":'/netpool/work/gpu-3/users/malyalasa/New_folder/tf_records',
    #"dataset_dir":'/bigpool/projects/yao_SCANGAN360/New_Folder/tf_records',
    "dataset_dir":'/bigpool/projects/yao_SCANGAN360/New_Folder/tf_records_w_costmap_dist_dir',
    #------------ Model Hyper paramaters ----------#
    "epochs":50,
    "lr":0.02,
    "optimizer":'nadam',
    "metric":'mae',#MeanAbsoluteError(),
    "losses":[euclidean_distance_loss],#Using directly during model compilation file, not used
    "loss_weights":[1],
    #----------- directory paths -----------------#
    "log_dir": '/netpool/work/gpu-3/users/malyalasa/New_folder/rosbag2numpy/logging',


    #---------- Model params -----------#
    "full_skip":True,
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

