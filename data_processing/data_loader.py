import tensorflow as tf
import pprint
from glob import glob
from typing import List, Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
import numpy as np

class dataset_loader:
    """Dataset loader class, fetch records, shuffle and split and build tf.data.Dataset"""

    def __init__(
        self, tfrec_dir: Path, train_size: float = 0.7, batch_size: int = 16, shuffle_buffer: int=32*10,
        normalize_coords:bool=True, normalize_factor=1
    ) -> None:
        self.autotune = tf.data.experimental.AUTOTUNE
        self.tfrecords_dir = tfrec_dir
        self.train_size = train_size
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.normalize_coords = normalize_coords 
        self.normalize_factor = normalize_factor
        

    # function to contruct back from tf record example
    def __prepare_sample(self, example):
        """contruct dictionary of features as keys and respective data as values

        Args:
            example ([type]): Dictiornary of data inputs and outputs- names as keys and actual data as values

        Returns:
            [type]: [description]
        """
        example["grid_map"] = tf.io.decode_raw(example["grid_map"], out_type=tf.int8)
        example["grid_map"] = tf.reshape(example["grid_map"], [1536, 1536])

        example["grid_org_res"] = tf.io.decode_raw(
            example["grid_org_res"], out_type=tf.float32
        )
        # example["grid_org_res"] = tf.reshape(example["grid_org_res"],[1,3])

        example["left_bnd"] = tf.io.decode_raw(example["left_bnd"], out_type=tf.float32)
        example["left_bnd"] = tf.reshape(example["left_bnd"], [25, 2])

        example["right_bnd"] = tf.io.decode_raw(
            example["right_bnd"], out_type=tf.float32
        )
        example["right_bnd"] = tf.reshape(example["right_bnd"], [25, 2])

        example["car_odo"] = tf.io.decode_raw(example["car_odo"], out_type=tf.float32)
        # example["car_odo"] = tf.reshape(example["car_odo"],[1,3])

        example["init_path"] = tf.io.decode_raw(
            example["init_path"], out_type=tf.float32
        )
        example["init_path"] = tf.reshape(example["init_path"], [25, 2])

        example["opt_path"] = tf.io.decode_raw(example["opt_path"], out_type=tf.float32)
        example["opt_path"] = tf.reshape(example["opt_path"], [25, 2])

        # return format (input1,input2,input3, input4.......),output
        # order important
        
        if self.normalize_coords==True:
            nf = tf.constant(self.normalize_factor,dtype=tf.float32,name="normalization factor")
            example["left_bnd"] = ((example["left_bnd"]-example["grid_org_res"][:2])/example["grid_org_res"][2])/nf
            example["right_bnd"] = ((example["right_bnd"]-example["grid_org_res"][:2])/example["grid_org_res"][2])/nf
            example["init_path"] =((example["init_path"]-example["grid_org_res"][:2])/example["grid_org_res"][2])/nf
            example["opt_path"] = ((example["opt_path"]-example["grid_org_res"][:2])/example["grid_org_res"][2])/nf

            # we shouldnt normalize heading in ego position
            car_odo_norm = (example["car_odo"][:2]-example["grid_org_res"][:2])/example["grid_org_res"][2]
            head = example["car_odo"][2]
            example["car_odo"] = tf.squeeze(tf.concat([tf.reshape(car_odo_norm,shape=(2,1)),tf.reshape(head,shape=(1,1))],axis=0))

        
        return (example["grid_map"],example["grid_org_res"],example["left_bnd"],
            example["right_bnd"],example["car_odo"],example["init_path"],
            example["file_details"]),example["opt_path"]

    def __parse_tfrecord_fn_ds(self, example):
        """serialize data, using features dictionary

        Args:
            example ([type]): serialized single sample

        Returns:
            Dict: A dict mapping feature keys to Tensor
        """
        # Dict of features and their description
        feature_description = {
            # model inputs
            "grid_map": tf.io.FixedLenFeature([], tf.string),
            "grid_org_res": tf.io.FixedLenFeature([], tf.string),
            "left_bnd": tf.io.FixedLenFeature([], tf.string),
            "right_bnd": tf.io.FixedLenFeature([], tf.string),
            "car_odo": tf.io.FixedLenFeature([], tf.string),
            "init_path": tf.io.FixedLenFeature([], tf.string),
            "file_details" : tf.io.FixedLenFeature([],tf.string),
            # model ouput
            "opt_path": tf.io.FixedLenFeature([], tf.string),
        }

        # Parse a single Example proto
        example = tf.io.parse_single_example(example, feature_description)
        return example

    def get_dataset(self, filenames: List[Path], batch_size: int, data_set_name: str):
        """Get data set built usinf tf.data

        Args:
            filenames (List[Path]): paths of tfrec files to create resepctive data set (train, valid, test)
            batch_size (int): Batch size
            data_set_name (str): Name of the data set to be built, train/validation/test

        Returns:
            tf.data.TFRecordDataset: tf record dataset
        """
        try:
            assert data_set_name in ("train", "valid", "test")
            dataset = (
                tf.data.TFRecordDataset(filenames, num_parallel_reads=self.autotune)
                .map(self.__parse_tfrecord_fn_ds, num_parallel_calls=self.autotune)
                .map(self.__prepare_sample, num_parallel_calls=self.autotune)
            )

            if data_set_name == "train":
                dataset = dataset.shuffle(self.shuffle_buffer)

            dataset = dataset.batch(batch_size).prefetch(self.autotune)

        except (AssertionError, BaseException) as Err:
            print(f"Build failed, Reason:{Err}")

        return dataset

    def shuffle_nd_split(
        self, list_paths: List[Path], train_size: float = 0.7
    ) -> Tuple[List[Path], List[Path], List[Path]]:
        """shuffle and split tfrecords to prepare for building datasets

        Args:
            list_paths (List[Path]): List of paths of tf records
            train_size (float, optional): Training data set size, between 0 and 1 . Defaults to 0.7.

        Returns:
            Tuple[List[Path], List[Path], List[Path]]: list of train dataset paths, validation dataset paths, testing dataset paths
        """
        list_train, list_test = train_test_split(list_paths, test_size=1 - train_size, random_state=2021)
        list_test, list_valid = train_test_split(list_test, test_size=0.5, random_state=2021)

        print(f"Number of records, Train files:{len(list_train)}, validation files:{len(list_valid)}, Test Files:{len(list_test)}")

        return list_train, list_valid, list_test

    def build_dataset(self):
        """Build three datasets for training, validation and testing purposes

        Returns:
            train, validation, test datasets: tuple of three
        """
        # min requirement python 3.5
        tfrec_paths = glob(f"{self.tfrecords_dir}/**/*.tfrec", recursive=True)

        ds_train, ds_valid, ds_test = self.shuffle_nd_split(
            list_paths=tfrec_paths, train_size=self.train_size
        )

        print(f"Building Dataset.......\n")
        ds_train = self.get_dataset(
            ds_train, batch_size=self.batch_size, data_set_name="train"
        )
        ds_test = self.get_dataset(
            ds_test, batch_size=self.batch_size, data_set_name="test"
        )
        ds_valid = self.get_dataset(
            ds_valid, batch_size=self.batch_size, data_set_name="valid"
        )

        pprint.pprint(ds_train.element_spec, width=1)

        return ds_train, ds_valid, ds_test

    def __collect_tfrec_paths(self,list_dir):
        assert len(list_dir)>=1
        if len(list_dir)>1 :
            list_in_list= [glob(f"{fold_path}/**/*.tfrec", recursive=True) for fold_path in list_dir]
            flat_list = (np.concatenate(list_in_list).tolist())
        else:

            flat_list=glob(f"{list_dir[0]}/**/*.tfrec", recursive=True)
        return flat_list

    def build_scenario_dataset(self,consider_scenes:int,no_train_scene:int,no_valid_scene:int,no_test_scene:int):
        scenarios = sorted(glob(f"{self.tfrecords_dir}/*"))
        print(f"Scenarios:{scenarios}")
        #if not consider_scenes==len(scenarios):
        #scenarios = scenarios[0:consider_scenes]
        random.Random(2020).shuffle(scenarios)

        try:
            assert no_train_scene+no_valid_scene+no_test_scene==len(scenarios)
            
            list_train = scenarios[0:(no_train_scene-1)]
            list_valid = scenarios[no_train_scene:(no_train_scene+no_valid_scene)]
            list_test = scenarios[no_train_scene+no_valid_scene:(no_train_scene+no_valid_scene+no_test_scene)]
            
            """
            list_train = scenarios[2:]
            list_valid = [scenarios[1]]
            list_test = [scenarios[0]]
            """

            train_tfrecs = self.__collect_tfrec_paths(list_train)
            valid_tfrecs = self.__collect_tfrec_paths(list_valid)
            test_tfrecs = self.__collect_tfrec_paths(list_test)

            print(f"Building Dataset.......\n")

            ds_train = self.get_dataset(
                train_tfrecs, batch_size=self.batch_size, data_set_name="train"
            )
 
            ds_valid = self.get_dataset(
                valid_tfrecs, batch_size=self.batch_size, data_set_name="valid"
            )

            ds_test = self.get_dataset(
                test_tfrecs, batch_size=self.batch_size, data_set_name="test"
            )
            print(f"Number of records, Train files:{len(train_tfrecs)}, validation files:{len(valid_tfrecs)}, Test Files:{len(test_tfrecs)}")

            pprint.pprint(ds_train.element_spec, width=1)
            #glob(f"{self.tfrecords_dir}/**/*.tfrec", recursive=True)
        except AssertionError as Err:
            print(f"Build scenario dataset failed, {Err}")

        return ds_train, ds_valid, ds_test


if __name__=='__main__':
    pass