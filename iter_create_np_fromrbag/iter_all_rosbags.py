from genericpath import isdir, isfile
from class_np_maker import np_maker as np_m
from bag_reader import Read_Ros_Bag
import tensorflow as tf
import os
from fnmatch import fnmatch
from typing import List 
import logging
import numpy as np

# Logging setup
logging.basicConfig(filename='iter_bag2np.log', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
ch= logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s- %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# path details- rosbags base bath, target path , max length of initial path
root_path = r'D:\sd_dynObj'
file_pattern = '*.bag'
target_path = r'D:\numpy_sd_dynObj'
_MAX_LENGTH =25


def get_all_bag_paths(root:str,pattern:str)->List[str]:
    list_all_bags=[]
    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name, pattern):
                list_all_bags.append((os.path.join(path, name)))

    return list_all_bags

all_bag_paths = get_all_bag_paths(root=root_path,pattern=file_pattern)


def save_np():
    for i in range(len(all_bag_paths)):
        # read rosbag
        try:
            rosbag_reader = Read_Ros_Bag(path=all_bag_paths[i])
        
            data_params = rosbag_reader.msgs()

            # converting data to np arrays for easy manipulation
            convert_2_np = np_m(_MAX_LENGTH,**data_params)

            np_init_path, np_opt_path = convert_2_np.create_np_path()

            np_grid_data = convert_2_np.create_np_grid()

            np_image_data = convert_2_np.create_np_img()

            #create folder to save np arrays
            assert len(np_grid_data)==len(np_init_path)==len(np_opt_path)==len(np_image_data)
            
            if len(np_grid_data)>=10:
                no_samples = len(np_grid_data)
                folder_name = all_bag_paths[i].split("\\")[-2]
                np_file_name = all_bag_paths[i].split("\\")[-1].split(".")[0]

                if not isdir(os.path.join(target_path,folder_name)):
                    print("Directory exists")
                    os.makedirs(os.path.join(target_path,folder_name))
                    
                #np.save(os.path.join(target_path,folder_name,np_file_name+f"_nos{no_samples}"+"_grid"),np_grid_data)
                #np.save(os.path.join(target_path,folder_name,np_file_name+f"_nos{no_samples}"+"_init_path"),np_init_path)
                #np.save(os.path.join(target_path,folder_name,np_file_name+f"_nos{no_samples}"+"_opt_path"),np_opt_path)
                np.save(os.path.join(target_path,folder_name,np_file_name+f"_nos{no_samples}"+"_img"),np_image_data)
                logger.info(f"Saved Files of {np_file_name} from folder {folder_name}")
                
        except:
            logger.error(f"Error handling the file {all_bag_paths[i]}")

        
save_np()
