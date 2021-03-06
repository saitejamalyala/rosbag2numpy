from genericpath import isdir
from class_np_maker import np_maker as np_m
from bag_reader import Read_Ros_Bag
import time
import os
from fnmatch import fnmatch
from typing import List
from pathlib import Path
import logging
import numpy as np
import asyncio


# path details- rosbags base bath, target path , max length of initial path
root_path = r"D:\teja"
file_pattern = "*.bag"
target_path = r"D:\test"
_MAX_LENGTH = 25


# Logging setup
logging.basicConfig(filename="iter_bag2np.log", filemode="a", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s- %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


def get_all_bag_paths(root: Path, pattern: str) -> List[Path]:
    """Fetch list of all bag paths in the root directory (where all paths are stored)

    Args:
        root (Path): Root path
        pattern (str): pattern of file (file extension)

    Returns:
        List[Path]: List of paths of rosbags
    """   
    list_all_bags = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name, pattern):
                list_all_bags.append((os.path.join(path, name)))

    return list_all_bags

async def save_np(bag_path:Path):
    """save to compressed numpy arrays

    Args:
        bag_path (Path): path to rosbag file
    """
    # read rosbag
    try:
        rosbag_reader = Read_Ros_Bag(path=bag_path)

        data_params = rosbag_reader.msgs()

        # converting data to np arrays for easy manipulation
        convert_2_np = np_m(_MAX_LENGTH, **data_params)

        np_init_path, np_opt_path, np_bnd_path = convert_2_np.create_np_path()
        time.sleep(0.1)
        np_grid_data, np_all_grid_orig_res = convert_2_np.create_np_grid()
        time.sleep(0.1)
        np_image_data = convert_2_np.create_np_img()
        time.sleep(0.1)
        np_odo_data = convert_2_np.create_np_odo()

        # check if all lengths are equal
        assert (
            len(np_grid_data)
            == len(np_init_path)
            == len(np_opt_path)
            == len(np_bnd_path)
            == len(np_image_data)
            == len(np_odo_data)
        )

        # f len(np_grid_data)>=10:
        no_samples = len(np_grid_data)
        scenario_name = bag_path.split(os.sep)[-3]
        folder_name = bag_path.split(os.sep)[-2]
        np_file_name = bag_path.split(os.sep)[-1].split(".")[0]

        if not isdir(os.path.join(target_path, scenario_name, folder_name)):
            os.makedirs(os.path.join(target_path, scenario_name, folder_name))

        # Saving in numpy compressed format
        np.savez_compressed(
            os.path.join(
                target_path,
                scenario_name,
                folder_name,
                np_file_name + f"_nos{no_samples}" + "_grid",
            ),
            grid_data = np_grid_data,
            grid_org_res = np_all_grid_orig_res,
        )

        np.savez_compressed(
            os.path.join(
                target_path,
                scenario_name,
                folder_name,
                np_file_name + f"_nos{no_samples}" + "_init_path",
            ),
            init_path=np_init_path,
        )

        np.savez_compressed(
            os.path.join(
                target_path,
                scenario_name,
                folder_name,
                np_file_name + f"_nos{no_samples}" + "_opt_path",
            ),
            opt_path=np_opt_path,
        )

        np.savez_compressed(
            os.path.join(
                target_path,
                scenario_name,
                folder_name,
                np_file_name + f"_nos{no_samples}" + "_lr_bnd",
            ),
            left_bnd=np_bnd_path[:,0,:,:],
            right_bnd = np_bnd_path[:,1,:,:],
        )

        np.savez_compressed(
            os.path.join(
                target_path,
                scenario_name,
                folder_name,
                np_file_name + f"_nos{no_samples}" + "_img",
            ),
            image_data=np_image_data,
        )

        np.savez_compressed(            
            os.path.join(
                target_path,
                scenario_name,
                folder_name,
                np_file_name + f"_nos{no_samples}" + "_odo",
            ),
            odo_data=np_odo_data,
            )

        logger.info(f"Saved Files of {np_file_name} from folder {folder_name}")
        time.sleep(0.1)

    except BaseException as E:
        logger.error(f"Error handling the file {bag_path}, because of {E}")

async def main():
    """Loop through all bags and store necessary data into a compressex numpy arrray
    """
    all_bag_paths = get_all_bag_paths(root=root_path, pattern=file_pattern)

    for i in range(len(all_bag_paths)):
        await save_np(all_bag_paths[i])

asyncio.run(main())


