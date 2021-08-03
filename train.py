import tensorflow as tf
from glob import glob
from typing import List, Tuple
from .data_processing.data_loader import dataset_loader
from matplotlib import pyplot as plt
import numpy as np
from numpy import ndarray
from typing import Dict, List, Union
from .config import params
from .config import generalization_model_params as g_params
import wandb
from wandb.keras import WandbCallback
from .losses import euclidean_distance_loss,endpoint_loss
from .models import base_model,endpoint_in_model,generalizing_endpoint_model
from .models import conv1x1_endpoint_in_model,coordconv1x1_endpoint_in_model, LSTMconv1x1_endpoint_in_model 
import os
print(tf.__version__)
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"


def _get_optimizer(opt_name: str = "adam", lr: float = 0.02):

    if opt_name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        return tf.keras.optimizers.RMSprop(learning_rate=lr)


def _get_test_ds_size(ds_test) -> int:
    """get the size of test dataset

    Args:
        ds_test (tf.data.Dataset): [description]

    Returns:
        [int]: [Number of Samples inside dataset]
    """
    num_of_samples = 0
    # Looping through all batches in test dataset
    for input_batch, output_batch in ds_test:
        # Looping through all samples for a single (current) batch
        for i in range(0, len(input_batch[0].numpy())):
            num_of_samples += 1
    return num_of_samples


def get_np_test_ds(ds_test) -> Dict[str, Union[ndarray, List]]:
    """Test dataset (in tf.data.Dataset build) to numpy arrays

    Args:
        ds_test ([type]): tf.data.Dataset

    Returns:
        Dict[str,Union[ndarray,List]]: dictionary of gridmap,grid_org_res,left_bnd,right_bnd,car_odo,init_path,list_tst_file_details,opt_path
    """
    samples = _get_test_ds_size(ds_test)
    np_tst_gridmap = np.zeros(shape=(samples, 1536, 1536))
    np_tst_grid_org_res = np.zeros(shape=(samples, 3))
    np_tst_left_bnd = np.zeros(shape=(samples, 25, 2))
    np_tst_right_bnd = np.zeros(shape=(samples, 25, 2))
    np_tst_car_odo = np.zeros(shape=(samples, 3))
    np_tst_init_path = np.zeros(shape=(samples, 25, 2))
    np_tst_opt_path = np.zeros(shape=(samples, 25, 2))
    list_tst_file_details = []

    j = 0
    for input_batch, output_batch in ds_test:

        for i in range(0, len(input_batch[0].numpy())):
            np_tst_gridmap[j] = input_batch[0][i].numpy()
            np_tst_grid_org_res[j] = input_batch[1][i].numpy()
            np_tst_left_bnd[j] = input_batch[2][i].numpy()
            np_tst_right_bnd[j] = input_batch[3][i].numpy()
            np_tst_car_odo[j] = input_batch[4][i].numpy()
            np_tst_init_path[j] = input_batch[5][i].numpy()
            list_tst_file_details.append(input_batch[6][i])

            np_tst_opt_path[j] = output_batch[i].numpy()
            j = j + 1

    np_ds_test = {
        "grid_map": np_tst_gridmap,
        "grid_org_res": np_tst_grid_org_res,
        "left_bnd": np_tst_left_bnd,
        "right_bnd": np_tst_right_bnd,
        "car_odo": np_tst_car_odo,
        "init_path": np_tst_init_path,
        "file_details": list_tst_file_details,
        "opt_path": np_tst_opt_path,
    }

    # np_tst_gridmap,np_tst_grid_org_res,np_tst_left_bnd,np_tst_right_bnd,np_tst_car_odo,np_tst_init_path,list_tst_file_details,np_tst_opt_path
    return np_ds_test

class cd_wand_custom(WandbCallback):
    def __init__(
        self,
        # newly added
        ds_test,
        np_test_dataset:Dict[str,Union[ndarray,List]],
        test_index:int=15,
        normalized_coords=True,
        normalize_factor=1.0,
        # old
        monitor="val_loss",
        verbose=0,
        mode="auto",
        save_weights_only=False,
        log_weights=False,
        log_gradients=False,
        save_model=True,
        training_data=None,
        validation_data=None,
        labels=[],
        data_type=None,
        predictions=36,
        generator=None,
        input_type=None,
        output_type=None,
        log_evaluation=False,
        validation_steps=None,
        class_colors=None,
        log_batch_frequency=None,
        log_best_prefix="best_",
        save_graph=True,
        validation_indexes=None,
        validation_row_processor=None,
        prediction_row_processor=None,
        infer_missing_processors=True,
    ):
        super().__init__(
            monitor=monitor,
            verbose=verbose,
            mode=mode,
            save_weights_only=save_weights_only,
            log_weights=log_weights,
            log_gradients=log_gradients,
            save_model=save_model,
            training_data=training_data,
            validation_data=validation_data,
            labels=labels,
            data_type=data_type,
            predictions=predictions,
            generator=generator,
            input_type=input_type,
            output_type=output_type,
            log_evaluation=log_evaluation,
            validation_steps=validation_steps,
            class_colors=class_colors,
            log_batch_frequency=log_batch_frequency,
            log_best_prefix=log_best_prefix,
            save_graph=save_graph,
            validation_indexes=validation_indexes,
            validation_row_processor=validation_row_processor,
            prediction_row_processor=prediction_row_processor,
            infer_missing_processors=infer_missing_processors,
        )
        self.np_ds_test = np_test_dataset
        self.test_idx = test_index
        self.ds_test = ds_test
        self.normalized_coords = normalized_coords
        self.normalize_factor = normalize_factor
    pass

    def __get_index_data(self, np_ds_test, test_idx):
        test_data = {}
        test_data["grid_map"] = np_ds_test["grid_map"][test_idx]
        test_data["grid_org_res"] = np_ds_test["grid_org_res"][test_idx]
        test_data["left_bnd"] = np_ds_test["left_bnd"][test_idx]
        test_data["right_bnd"] = np_ds_test["right_bnd"][test_idx]
        test_data["car_odo"] = np_ds_test["car_odo"][test_idx]
        test_data["init_path"] = np_ds_test["init_path"][test_idx]
        test_data["opt_path"] = np_ds_test["opt_path"][test_idx]
        test_data["file_details"] = np_ds_test["file_details"][test_idx]
        test_data["testidx"] = test_idx

        test_data["predictions"] = np_ds_test["predictions"][test_idx]

        return test_data

    def __plot_scene(self, epoch, features):
        grid_map = features["grid_map"]
        grid_org = features["grid_org_res"]  # [x,y,resolution]
        left_bnd = features["left_bnd"]
        right_bnd = features["right_bnd"]
        init_path = features["init_path"]
        opt_path = features["opt_path"]
        car_odo = features["car_odo"]

        predict_path = features["predictions"]
        file_details = features["file_details"]

        plt.figure(figsize=(10, 10), dpi=200)

        if self.normalized_coords:
            # ax=fig.add_subplot(1,1,1)
            plt.plot(
                left_bnd[:, 0]/self.normalize_factor,
                left_bnd[:, 1]/self.normalize_factor,
                "-.",
                color="magenta",
                markersize=0.5,
                linewidth=0.5,
            )

            plt.plot(
                init_path[:, 0]/self.normalize_factor,
                init_path[:, 1]/self.normalize_factor,
                "o-",
                color="lawngreen",
                markersize=1,
                linewidth=1,
            )
            plt.plot(
                opt_path[:, 0]/self.normalize_factor,
                opt_path[:, 1]/self.normalize_factor,
                "--",
                color="yellow",
                markersize=1,
                linewidth=1,
            )

            plt.plot(
                predict_path[:, 0]/self.normalize_factor,
                predict_path[:, 1]/self.normalize_factor,
                "--",
                color="orange",
                markersize=1,
                linewidth=1,
            )

            plt.plot(
                right_bnd[:, 0]/self.normalize_factor,
                right_bnd[:, 1]/self.normalize_factor,
                "-.",
                color="magenta",
                markersize=0.5,
                linewidth=0.5,
            )

            plt.plot(
                car_odo[0]/self.normalize_factor,
                car_odo[1]/self.normalize_factor,
                "r*",
                color="red",
                markersize=8,
            )

        else:
            # print(type(grid_map))
            # ax=fig.add_subplot(1,1,1)
            res = grid_org[2]
            plt.plot(
                (left_bnd[:, 0] - grid_org[0]) / res,
                (left_bnd[:, 1] - grid_org[1]) / res,
                "-.",
                color="magenta",
                markersize=0.5,
                linewidth=0.5,
            )

            plt.plot(
                (init_path[:, 0] - grid_org[0]) / res,
                (init_path[:, 1] - grid_org[1]) / res,
                "o-",
                color="lawngreen",
                markersize=1,
                linewidth=1,
            )
            plt.plot(
                (opt_path[:, 0] - grid_org[0]) / res,
                (opt_path[:, 1] - grid_org[1]) / res,
                "--",
                color="yellow",
                markersize=1,
                linewidth=1,
            )

            plt.plot(
                (predict_path[:, 0] - grid_org[0]) / res,
                (predict_path[:, 1] - grid_org[1]) / res,
                "--",
                color="orange",
                markersize=1,
                linewidth=1,
            )

            plt.plot(
                (right_bnd[:, 0] - grid_org[0]) / res,
                (right_bnd[:, 1] - grid_org[1]) / res,
                "-.",
                color="magenta",
                markersize=0.5,
                linewidth=0.5,
            )

            plt.plot(
                (car_odo[0] - grid_org[0]) / res,
                (car_odo[1] - grid_org[1]) / res,
                "r*",
                color="red",
                markersize=8,
            )

        plt.legend(
            [
                "Left bound",
                "gt_init_path",
                "gt_opt_path",
                "predicted_path",
                "right bound",
                "car_centre",
            ],
            loc="lower left",
        )

        plt.imshow(grid_map, origin="lower")

        plt.title(f"{file_details}\nTest Index: {features['testidx']}")
        # save_fig_dir = '/netpool/work/gpu-3/users/malyalasa/New_folder/rosbag2numpy/test_results'
        # fig.savefig(f"{save_fig_dir}/Test_index_{features['testidx']}.jpg",format='jpg',dpi=300)
        # print(type(file_details))
        # cp_plt = plt
        wandb.log({f"sample_img_{epoch-1}": plt})
        plt.close()

        return plt

    def on_epoch_begin(self, epoch, logs):

        np_predictions = self.model.predict(self.ds_test)
        self.np_ds_test["predictions"] = np_predictions
        sample_data = self.__get_index_data(
            np_ds_test=self.np_ds_test, test_idx=self.test_idx
        )

        sample_fig = self.__plot_scene(epoch,features=sample_data)
        """
        if (epoch-1) % 2 == 0:
            wandb.log({f"sample_img_{epoch-1}": sample_fig})
            sample_fig.close()
        """
        return super().on_epoch_begin(epoch, logs=logs)

    def on_train_end(self, logs):
        test_loss, test_accuracy = self.model.evaluate(self.ds_test)
        wandb.log({"test_loss":test_loss,"test_accuracy":test_accuracy})
        return super().on_train_end(logs=logs)

if __name__ == "__main__":
    wandb.init(project="ppmodel_base", config=params)

    # Load dataset
    ds_loader = dataset_loader(
        tfrec_dir=params.get("dataset_dir"),
        batch_size=params.get("H_BATCH_SIZE"),
        shuffle_buffer=params["H_SHUFFLE_BUFFER"],
        normalize_coords=params["normalize_coords"],
    )

    #ds_train, ds_valid, ds_test = ds_loader.build_dataset()
    ds_train, ds_valid, ds_test = ds_loader.build_scenario_dataset(consider_scenes=10,no_train_scene=8,no_valid_scene=1,no_test_scene=1)

    np_ds_test = get_np_test_ds(ds_test=ds_test)

    # Build and compile model
    #pp_model = base_model.nn()
    #pp_model = endpoint_in_model.nn(full_skip=params.get("full_skip"))
    #pp_model = generalizing_endpoint_model.nn(full_skip= g_params.get("full_skip"))
    #pp_model = conv1x1_endpoint_in_model.nn(full_skip= params.get("full_skip"),params=params)

    #strategy = tf.distribute.MirroredStrategy()
    #print(f'Number of replicas in sync {strategy.num_replicas_in_sync}')

    #with strategy.scope():
        #pp_model = LSTMconv1x1_endpoint_in_model.nn(full_skip= params.get("full_skip"),params=params)
    pp_model = coordconv1x1_endpoint_in_model.nn(full_skip= params.get("full_skip"),params=params)

    opt = _get_optimizer(params.get("optimizer"), lr=params.get("lr"))
    pp_model.compile(
        optimizer=opt, 
        loss=params.get("losses"),
        loss_weights=params.get("loss_weights"), metrics=params.get("metric")
    )
        
    # Learning rate scheduler
    cb_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=6, min_lr=0.0001
    )

    #tensorboardcallback = tf.keras.callbacks.TensorBoard(log_dir=params.get("log_dir"),profile_batch=2)

    # Model training
    history = pp_model.fit(
        ds_train,
        epochs=params.get("epochs"),
        validation_data=ds_valid,
        callbacks=[
            cb_reduce_lr,
            #WandbCallback(),
            cd_wand_custom(ds_test=ds_test, np_test_dataset=np_ds_test,test_index=40,normalized_coords=params.get("normalize_coords"))
        ],
    )

    #test_loss, test_accuracy = pp_model.evaluate(ds_test)