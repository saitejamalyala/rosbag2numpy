params = {
    "Description": 'checking previous run with updated get_config, as i wasnt able to load model, because of wrong config',

    #-------------- Data set Parameters ------------#
    "H_BATCH_SIZE" :32,
    "H_SHUFFLE_BUFFER" : 32*100,

    "dataset_dir":'/netpool/work/gpu-3/users/malyalasa/New_folder/tf_records',
    #------------ Model Hyper paramaters ----------#
    "epochs":50,
    "lr":0.02,
    "optimizer":'adam',
    "metric":'accuracy',
    "loss_weights":[1,0],
    #----------- directory paths -----------------#
    "log_dir": '/netpool/work/gpu-3/users/malyalasa/New_folder/rosbag2numpy/logging',
    
}


