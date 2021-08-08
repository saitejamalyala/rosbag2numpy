import numpy as np
from numpy.core.numeric import array_equiv
import tensorflow as tf
from typing import List, Dict, Tuple, Type
from numpy import ndarray
from pathlib import Path
from fnmatch import fnmatch
import re
import logging
import os
from tqdm import trange
import numpy as np

print(tf.__version__)

# Logging setup
logging.basicConfig(filename="nzp2tfrec-zeros_check.log", filemode="a", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s- %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# Class
class Tfrecs_frm_npz:
    """Create tensorflow records from compressed numpy arrays"""

    def __init__(self, source_path, target_path, samples_per_record: int = 16) -> None:
        self.source_path = source_path
        self.target_path = target_path
        self.num_samples = samples_per_record  # samples per record

        # file patterns to load data from
        self.file_patterns = [
            "*_grid.npz",
            "*_init_path.npz",
            "*_opt_path.npz",
            "*_lr_bnd.npz",
            "*_odo.npz",
        ]
        pass

    @staticmethod
    def __get_all_paths(root: Path, pattern: str) -> List[Path]:
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

    def __get_all_msg_paths(self) -> Tuple[List[Path]]:
        """get all paths for the specified file patterns

        Returns:
            Tuple[List[Path]]: grid_paths,init_paths,opt_paths,lr_bnd_paths,odo_paths
        """
        list_all_paths = [
            self.__get_all_paths(self.source_path, pattern=fp)
            for fp in self.file_patterns
        ]

        grid_paths, init_paths, opt_paths, lr_bnd_paths, odo_paths = list_all_paths

        """
        (
            grid_paths = self.__get_all_paths(self.source_path, pattern="*_grid.npz")
            init_paths = self.__get_all_paths(self.source_path, pattern="*_init_path.npz")
            opt_paths = self.__get_all_paths(self.source_path, pattern="*_opt_path.npz")
            lr_bnd_paths = self.__get_all_paths(self.source_path, pattern="*_lr_bnd.npz")
            odo_paths = self.__get_all_paths(self.source_path, pattern="*_odo.npz")
        )
        """

        try:
            assert (
                len(grid_paths)
                == len(init_paths)
                == len(opt_paths)
                == len(lr_bnd_paths)
                == len(odo_paths)
            )
        except AssertionError:
            print(
                f"Assertion error, all inputs and outputs are not of equal length [number of samples]"
            )

        return grid_paths, init_paths, opt_paths, lr_bnd_paths, odo_paths

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
    def __assert_lengths(in1, in2, in3, in4, in5, in6, in7):
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
            assert (
                len(in1)
                == len(in2)
                == len(in3)
                == len(in3)
                == len(in4)
                == len(in5)
                == len(in6)
                == len(in7)
            )
        except AssertionError:
            print(f"Assertion error, all inputs and outputs are not of equal length")

    @staticmethod
    def __create_example(
        grid_map: ndarray,
        grid_org_res: ndarray,
        left_bnd: ndarray,
        right_bnd: ndarray,
        car_odo: ndarray,
        init_path: ndarray,
        opt_path: ndarray,
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
            value = calc_costmap(griddata=value,ododata=car_odo,initial_path=init_path)

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

        def calc_direction_cost(car_coords:ndarray,endpoint_unit_normal:ndarray,vector:ndarray):
            direction = car_coords-vector
            direction_unit = direction/np.linalg.norm(x=direction)

            direction_cost = np.dot(direction_unit,endpoint_unit_normal)

            return direction_cost

        def calc_costmap(gridmap:ndarray,car_odo:ndarray,initial_path:ndarray,include_dir_cost:bool=True,distance_mag:float=0.23)->ndarray:
            """Function to transfor grid map to cost map

            Args:
                gridmap (ndarray): Binary occupancy grid map
                car_odo (ndarray): [description]
                initial_path (ndarray): [description]
                distance_mag (float, optional): lower it is larger cost considered for the occupied cells. Defaults to 0.40.

            Returns:
                ndarray: [description]
            """
            
            car_coords = car_odo[0:2]
            end_coords = initial_path[-1]

            # Initialize distance and direction costmap 
            dist_costmap = np.zeros(shape=(1536,1536))
            dir_costmap =  np.zeros(shape=(1536,1536))

            ################# calculate distance costmap ################
            
            # get occuppied cell indices
            (y,x) = np.where(gridmap==1)
            # create coordinate pairs of location of occupied cells
            coords = np.array((x,y)).T

            # eucledian difference
            distance = np.square(coords-car_coords)
            distance = np.sum(distance, axis=1)
            distance = 1/(distance**distance_mag)

            dist_costmap[(y,x)]=distance

            ################# Direction cost map ######################
            
            if include_dir_cost:

                # end point direction
                direction_vector = car_coords-end_coords

                # unit vector between, car location and end point
                end_dir_unit = direction_vector/np.linalg.norm(x=direction_vector)

                # calculate dir cost for all occupied cell locations
                direction_cost= np.asarray([calc_direction_cost(car_coords=car_coords,endpoint_unit_normal=end_dir_unit,vector=coord) for coord in coords])

                # zero down all opposite directions to end point
                direction_cost[np.where(direction_cost<0)]=0.0

                dir_costmap[(y,x)] = direction_cost
            else:
                dir_costmap = np.ones(shape=(1536,1536))

            # pointwise multiply distance cost and direction cost
            costmap = dist_costmap*dir_costmap

            # normalize cost map between [0,1]
            costmap = costmap/np.max(costmap)

            return costmap

        feature = {
            # model inputs
            "grid_map": gridmap_feature(grid_map),
            "grid_org_res": bytes_feature(grid_org_res),
            "left_bnd": bytes_feature(left_bnd),
            "right_bnd": bytes_feature(right_bnd),
            "car_odo": bytes_feature(car_odo),
            "init_path": bytes_feature(init_path),
            "file_details":string_feature(file_details),
            # model outputs
            "opt_path": bytes_feature(opt_path),
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

        grid_map = file_data.get("in_grid_map")
        grid_org_res = file_data.get("in_grid_org_res")
        left_bnd = file_data.get("in_left_bnd")
        right_bnd = file_data.get("in_right_bnd")
        car_odo = file_data.get("in_car_odo")
        init_path = file_data.get("in_init_path")

        opt_path = file_data.get("out_opt_path")

        scene_name, folder_name, file_name, file_prefix = self.__get_names(grid_path)

        current_tfrec_dir = os.path.join(self.target_path, scene_name, folder_name)

        num_tfrecods = len(grid_map) // self.num_samples

        if len(grid_map) % self.num_samples:
            num_tfrecods += 1  # add one record if there are any remaining samples

        logger.info(
            f"file:{file_name} having {len(grid_map)} samples, creating {num_tfrecods} records"
        )

        if not os.path.exists(current_tfrec_dir):
            os.makedirs(current_tfrec_dir)

        for tfrec_num in range(num_tfrecods):
            samples = grid_map[
                (tfrec_num * self.num_samples) : ((tfrec_num + 1) * self.num_samples)
            ]

            with tf.io.TFRecordWriter(
                f"{current_tfrec_dir}/{file_prefix}_file_{tfrec_num:02d}-{len(samples)}.tfrec"
            ) as writer:
                for s in range(len(samples)):
                    # print(np.shape(np_init_path[i]))
                    # create example for single sample
                    # Order of the parameters passed in __create_example is critically important
                    if grid_org_res[s][2]==0.0 or grid_org_res[s][1]==0.0 or grid_org_res[s][0]==0.0:
                        logger.info(f"Files with zeros: {scene_name}/{folder_name}/{file_name}")
                    elif np.array_equiv(init_path[s],np.zeros_like(init_path[s])):
                        logger.info(f"Files with zeros: {scene_name}/{folder_name}/{file_name}")
                    elif np.array_equiv(left_bnd[s],np.zeros_like(left_bnd[s])):
                        logger.info(f"Files with zeros: {scene_name}/{folder_name}/{file_name}")
                    elif np.array_equiv(right_bnd[s],np.zeros_like(right_bnd[s])):
                        logger.info(f"Files with zeros: {scene_name}/{folder_name}/{file_name}")
                    else :
                        logger.info(f"Grid_org_res:{grid_org_res[s]} in {scene_name}/{folder_name}/{file_name}")

                    example = self.__create_example(
                        grid_map=grid_map[s],
                        grid_org_res=grid_org_res[s],
                        left_bnd=left_bnd[s],
                        right_bnd=right_bnd[s],
                        car_odo=car_odo[s],
                        init_path=init_path[s],
                        opt_path=opt_path[s],
                        file_details = bytes(f"{scene_name}/{folder_name}/{file_name}",encoding='utf-8')
                    )
                    # serialize the single sample example to string
                    writer.write(example.SerializeToString())

    def create_records(self):
        """Create records from paths of numpy arrays and write them to target directory"""
        (
            all_grid_paths,
            all_init_paths,
            all_opt_paths,
            all_lr_bnd_paths,
            all_odo_paths,
        ) = self.__get_all_msg_paths()

        # logger.info(len(all_grid_paths),len(all_init_paths),len(all_opt_paths),len(all_lr_bnd_paths),len(all_odo_paths))

        for i in trange(len(all_grid_paths[0:1])):
            # load inputs
            in_grid_map = self.__load_npz(
                file_path=all_grid_paths[i],
                array_name="grid_data",
                data_type=np.int8,
                normalize_factor=127.0,
            )
            in_grid_org_res = self.__load_npz(all_grid_paths[i], "grid_org_res")
            in_init_path = self.__load_npz(all_init_paths[i], "init_path")
            in_left_bnd = self.__load_npz(all_lr_bnd_paths[i], "left_bnd")
            in_right_bnd = self.__load_npz(all_lr_bnd_paths[i], "right_bnd")
            in_car_odo = self.__load_npz(all_odo_paths[i], "odo_data")
            
            # load outputs
            out_opt_path = self.__load_npz(all_opt_paths[i], "opt_path")

            self.__assert_lengths(
                in_grid_map,
                in_grid_org_res,
                in_init_path,
                out_opt_path,
                in_left_bnd,
                in_right_bnd,
                in_car_odo,
            )

            """
            logger.info(
                f"\n \
                Grid map:{np.shape(in_grid_map)}\n \
                Grid Orig resol:{np.shape(in_grid_org_res)}\n \
                Init path:{np.shape(in_init_path)}\n \
                Opt Path:{np.shape(out_opt_path)}\n \
                Left bnd:{np.shape(in_left_bnd)}\n \
                Right bnd{np.shape(in_right_bnd)}\n \
                Car x,y,theta:{np.shape(in_car_odo)}\n")
            """

            dict_data = {
                "in_grid_map": in_grid_map,
                "in_grid_org_res": in_grid_org_res,
                "in_left_bnd": in_left_bnd,
                "in_right_bnd": in_right_bnd,
                "in_car_odo": in_car_odo,
                "in_init_path": in_init_path,
                "out_opt_path": out_opt_path,
            }

            # write tf records
            self.__write_records(all_grid_paths[i], **dict_data)


s_path = r"D:\npz_files"
tgt_path = r"D:\tf_records"

tf_rec_maker = Tfrecs_frm_npz(
    source_path=s_path, target_path=tgt_path, samples_per_record=16
)

tf_rec_maker.create_records()
