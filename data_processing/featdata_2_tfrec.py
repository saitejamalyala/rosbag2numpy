import numpy as np
from numpy.core.fromnumeric import shape
from numpy.lib.function_base import diff
import tensorflow as tf
from typing import List, Dict, Tuple, Type
from numpy import ndarray
from pathlib import Path
from fnmatch import fnmatch
import re
import logging
import os
from tqdm import trange

print(tf.__version__)

# Logging setup
logging.basicConfig(filename="tfrec_costmap_fv.log", filemode="a", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s- %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

class FvNpz_2_TfRec:
    """Create tensorflow records from compressed numpy arrays"""

    def __init__(self, source_path:Path, target_path:Path, samples_per_record: int = 16) -> None:
        self.source_path = source_path
        self.target_path = target_path
        self.num_samples = samples_per_record  # samples per record

        # file patterns to load data from
        # 
        self.file_pattern = ['*cm_fv_paths_odo_unequal.npz']#['*bgm_fv_paths_odo_unequal.npz']#['*cm_fv_paths_odo_unequal.npz']#['*_costmap_init_opt_diff_odo_fv.npz']

        pass

    #@staticmethod
    def get_all_paths(self,root: Path, pattern: str)->List[Path]:
        """Walk through root directory to find paths for file with requested file patterns

        Args:
            root (str): path to root directory
            pattern (str): file pattern

        Returns:
            List[Path]: List of paths
        """
        list_file_paths = []
        for path, subdirs, files in os.walk(root):
            for file_name in files:
                if fnmatch(file_name, pattern):
                    list_file_paths.append((os.path.join(path, file_name)))

        return sorted(list_file_paths)


    @staticmethod
    def __load_npz(
        file_path: Path,
        array_name: str,
        data_type:Type =np.float32,
        normalize_factor: float = 1.0,
    ) -> ndarray:
        """To load compressed numpy file and perform type casting anf normalization

        Args:
            file_path (str): Path to the compressed numpy array
            array_name (str): array name used to create the numpy array
            data_type ([type], optional): type of the data type. Defaults to np.float32.
            normalize_factor (float, optional): Normalization factor. Defaults to 1.0.

        Returns:
            ndarray: loaded numpy array, normalized and type casted
        """
        npz_array = np.load(file_path)[array_name]
        npz_array = (npz_array / normalize_factor).astype(data_type)

        return npz_array

    @staticmethod
    def __get_names(path: Path) -> Tuple[str, str, str, str]:
        """Get Scenario name, folder name, file name, file_prefix

        Args:
            path (str): os.path like path

        Returns:
            Tuple[str,str,str,str]: scene_name, folder_name, file_name, file_prefix
        """
        list_split_path = path.split(os.sep)
        scene_name = list_split_path[-3]
        folder_name = list_split_path[-2]
        file_name = list_split_path[-1]
        file_prefix = re.split("_nos", file_name)[0]

        return scene_name, folder_name, file_name, file_prefix

    @staticmethod
    def __assert_lengths(in1, in2, in3, in4, in5, in6):
        """[summary]

        Args:
            in1 ([type]): [description]
            in2 ([type]): [description]
            in3 ([type]): [description]
            in4 ([type]): [description]
            in5 ([type]): [description]
            in6 ([type]): [description]
            in7 ([type]): [description]
        """
        try:
            #logger.info(f'{in1.shape[0]},{in2.shape[0]},{in3.shape[0]},{in4.shape[0]},{in5.shape[0]},{in6.shape[0]}')
            assert (
                len(in1)
                == len(in2)
                == len(in3)
                == len(in3)
                == len(in4)
                == len(in5)
                == len(in6)
            )
        except AssertionError:
            print(f"Assertion error, all inputs and outputs are not of equal length")

    @staticmethod
    def __create_example(
        costmap: ndarray,
        init_path: ndarray,
        feature_vector: ndarray,
        opt_path: ndarray,
        car_odo: ndarray,
        unequal_indices: ndarray,
        diff_path: ndarray,
        file_details:str,
    ) -> tf.train.Example:
        """To create a tf.train.Example for sample

        Args:
            grid_map (ndarray): Occupancy grid map with shape(1536,1536)
            grid_org_res (ndarray): Occupancy grid map origin and resolution-[x,y,resolution] with shape (,3)
            left_bnd (ndarray): Left boundary with shape(25,2)
            right_bnd (ndarray): Right boundary with shape (25,2)
            car_odo (ndarray): Car centre co-ordinates and heading(orientation)-[x,y,theta] with shape (,3)
            init_path (ndarray): Initial path with shape (25,2)
            opt_path (ndarray): Optimal path with shape (25,2)

        Returns:
            tf.train.Example: Example with Features (Dict)
        """
        def gridmap_feature(value):
            """Returns a bytes_list from a string / byte."""

            return tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[value.tostring()])
            )

        def bytes_feature(value):
            """Returns a bytes_list from a string / byte."""
            return tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[value.tostring()])
            )
        
        def string_feature(value):
            """Returns a bytes_list from a string / byte."""
            return tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[value])
            )

        feature = {
            # model inputs
            "cost_map": gridmap_feature(costmap),
            "init_path": bytes_feature(init_path),
            "feature_vector": bytes_feature(feature_vector),
            "opt_path": bytes_feature(opt_path),
            "car_odo": bytes_feature(car_odo),
            "file_details":string_feature(file_details),
            "unequal_indices":bytes_feature(unequal_indices),
            # model outputs
            "diff_path":bytes_feature(diff_path)
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def __write_records(self, grid_path: Path, **file_data: Dict[str, ndarray]):
        """Write the data in ndarrays to tf records

        Args:
            grid_path (Path): grid data path to extract scene name, folder name,
             bag file name and file prefix(to be appened to tfrec file name).
             Can use other path also instead of grid_path

            file_data(Dict[str, ndarray]): Dictionary with names of the ndarrays as keys and ndarrays as values
        """

        costmap = file_data.get("ip_costmap")
        init_path = file_data.get("ip_init_path")
        feature_vector = file_data.get("ip_fv")
        opt_path = file_data.get("ip_opt_path")
        car_odo = file_data.get("ip_odo")
        unequal_indices = file_data.get("ip_unequal_path_indices")
        
        diff_path = file_data.get("op_diff_path")


        scene_name, folder_name, file_name, file_prefix = self.__get_names(grid_path)

        current_tfrec_dir = os.path.join(self.target_path, scene_name, folder_name)

        num_tfrecods = len(costmap) // self.num_samples

        if len(costmap) % self.num_samples:
            num_tfrecods += 1  # add one record if there are any remaining samples

        logger.info(
            f"file:{file_name} having {len(costmap)} samples, creating {num_tfrecods} records"
        )

        if not os.path.exists(current_tfrec_dir):
            os.makedirs(current_tfrec_dir)

        for tfrec_num in range(num_tfrecods):
            samples = costmap[
                (tfrec_num * self.num_samples) : ((tfrec_num + 1) * self.num_samples)
            ]

            with tf.io.TFRecordWriter(
                f"{current_tfrec_dir}/{file_prefix}_file_{tfrec_num:02d}-{len(samples)}.tfrec"
            ) as writer:
                for s in range(len(samples)):
                    # print(np.shape(np_init_path[i]))
                    # create example for single sample
                    # Order of the parameters passed in __create_example is critically important
                    if np.array_equiv(init_path[s],np.zeros_like(init_path[s])):
                        logger.info(f"Files with zeros: {scene_name}/{folder_name}/{file_name}")
                    elif np.array_equiv(opt_path[s],np.zeros_like(opt_path[s])):
                        logger.info(f"Files with zeros: {scene_name}/{folder_name}/{file_name}")
                    elif np.array_equiv(car_odo[s],np.zeros_like(car_odo[s])):
                        logger.info(f"Files with zeros: {scene_name}/{folder_name}/{file_name}")

                    example = self.__create_example(
                        costmap=costmap[s],
                        init_path=init_path[s],
                        feature_vector=feature_vector[s],
                        car_odo=car_odo[s],
                        opt_path=opt_path[s],
                        unequal_indices = unequal_indices,

                        diff_path = diff_path[s],
                        file_details = bytes(f"{scene_name}/{folder_name}/{file_name}",encoding='utf-8')
                    )
                    # serialize the single sample example to string
                    writer.write(example.SerializeToString())
    

    def create_records(self):


        # get all paths
        file_paths = self.get_all_paths(root=self.source_path, pattern=self.file_pattern[0])
  
        # iterate through every path
        for f in trange(len(file_paths)):

            #potential inputs
            ip_costmap = self.__load_npz(file_path=file_paths[f],array_name='costmap')
            #convert to uint8 to save disk space, divide by 255 when using
            ip_costmap = (ip_costmap*255).astype(np.uint8)

            ip_init_path = self.__load_npz(file_path=file_paths[f],array_name='init_path')
            ip_opt_path = self.__load_npz(file_path=file_paths[f],array_name='opt_path')
            ip_odo = self.__load_npz(file_path=file_paths[f],array_name='odo_data')
            ip_fv = self.__load_npz(file_path=file_paths[f],array_name='feature_vector')
            ip_unequal_path_indices = self.__load_npz(file_path=file_paths[f],array_name='unequal_path_idx')

            #output
            op_diff_path = self.__load_npz(file_path=file_paths[f],array_name='diff_path')

            self.__assert_lengths(
                ip_costmap,
                ip_init_path,
                ip_opt_path,
                ip_odo,
                ip_fv,
                op_diff_path,
            )


        # load all data corresponding to that file path
        # for every loaded data write records
            dict_data = {
                "ip_costmap":ip_costmap,
                "ip_init_path":ip_init_path,
                "ip_opt_path":ip_opt_path,
                "ip_odo":ip_odo,
                "ip_fv":ip_fv,
                "ip_unequal_path_indices":ip_unequal_path_indices,
                "op_diff_path":op_diff_path,
            }

            self.__write_records(file_paths[f], **dict_data)


        pass



if __name__ == "__main__":

    s_path = r"C:\Users\Teja\Documents\_INFOTECH\Thesis\sample_Ros_bag\np_data\raw_data_wo_img"
    tgt_path = r"D:\tf_records_w_bgm_fv_diff_unequal"#r"D:\tf_records_w_cm_fv_diff_unequal"
    #r"D:\tf_records_w_cm_fv_diff"
    #r"C:\Users\Teja\Documents\_INFOTECH\Thesis\sample_Ros_bag\np_data\tfrec_fv_costmap"

    tf_rec_maker = FvNpz_2_TfRec(
        source_path=s_path, target_path=tgt_path, samples_per_record=16
    )
    #print(len(tf_rec_maker.get_all_paths(root=s_path,pattern='*_costmap_init_opt_diff_odo_fv.npz')))
    tf_rec_maker.create_records()

