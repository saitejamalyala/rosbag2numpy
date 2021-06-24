import tensorflow as tf
from glob import glob
from typing import List,Tuple
from .data_processing.data_loader import dataset_loader
from matplotlib import pyplot as plt
import numpy as np
from .config import params
import wandb
from wandb.keras import WandbCallback
from .losses import euclidean_distance_loss
from .models import base_model


def _get_optimizer(opt_name:str='adam',lr:float=0.02):

    if opt_name=='adam':
        return tf.keras.optimizers.Adam(learning_rate=0.02)
    else:
        return tf.keras.optimizers.Adam(learning_rate=0.02)



if __name__=='__main__':
    wandb.init(project="ppmodel_base",config=params)

    ds_loader = dataset_loader(tfrec_dir=params.get("dataset_dir"),batch_size=params.get("H_BATCH_SIZE"))
    ds_train,ds_valid,ds_test= ds_loader.build_dataset()

    pp_model = base_model.nn()
    opt = _get_optimizer(params.get("optimizer"),lr=params.get("lr"))
    pp_model.compile(optimizer=opt,loss=euclidean_distance_loss, metrics=params.get("metric"))
    cb_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=4, min_lr=0.0001)
    
    
    history = pp_model.fit(ds_train,epochs=50,validation_data=ds_valid, callbacks=[cb_reduce_lr,WandbCallback()])

    test_loss,test_accuracy = pp_model.evaluate(ds_test)
