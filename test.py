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
import os

os.environ['CUDA_VISIBLE_DEVICES']="2,3"

def _get_optimizer(opt_name:str='adam',lr:float=0.02):

    if opt_name=='adam':
        return tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        return tf.keras.optimizers.Adam(learning_rate=lr)


ds_loader = dataset_loader(tfrec_dir=params.get("dataset_dir"),batch_size=params.get("H_BATCH_SIZE"))
ds_train,ds_valid,ds_test= ds_loader.build_dataset()

pp_model=tf.keras.models.load_model('/netpool/work/gpu-3/users/malyalasa/New_folder/wandb/run-20210625_053926-21xquqgr/files/model-best.h5',compile=False)

opt = _get_optimizer(params.get("optimizer"),lr=params.get("lr"))
pp_model.compile(optimizer=opt,loss=euclidean_distance_loss, metrics=params.get("metric"))

test_loss,test_accuracy = pp_model.evaluate(ds_test)

