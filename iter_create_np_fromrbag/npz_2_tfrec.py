from unicodedata import name
import numpy as np
import tensorflow as tf
from glob import glob
from typing import List,Dict, Tuple, Type
from fnmatch import fnmatch
import matplotlib.pyplot as plt
import re
import logging
import os
from tqdm import trange
print(tf.__version__)

# Logging setup
logging.basicConfig(filename=r".\nzp2tfrec.log", filemode="a", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s- %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

class Tfrecs_frm_npz():

    def __init__(self, source_path, target_path, samples_per_record:int=16) -> None:
        self.source_path = source_path
        self.target_path = target_path
        self.num_samples = samples_per_record # samples per record
        pass

    @staticmethod
    def __get_all_paths(root:str, pattern:str)->List[str]:
        list_file_paths=[]
        for path, subdirs, files in os.walk(root):
            for file_name in files:
                if fnmatch(file_name, pattern):
                    list_file_paths.append((os.path.join(path, file_name)))
        
        return sorted(list_file_paths)

    def __get_all_msg_paths(self)->Tuple[List[str]]:

        grid_paths = self.__get_all_paths(self.source_path,pattern="*_grid.npz")
        init_paths = self.__get_all_paths(self.source_path,pattern="*_init_path.npz")
        opt_paths = self.__get_all_paths(self.source_path,pattern="*_opt_path.npz")
        lr_bnd_paths = self.__get_all_paths(self.source_path,pattern="*_lr_bnd.npz")
        odo_paths = self.__get_all_paths(self.source_path,pattern="*_odo.npz")
        
        try:
            assert len(grid_paths)==len(init_paths)==len(opt_paths)==len(lr_bnd_paths)==len(odo_paths)
        except AssertionError:
            print(f"Assertion error, all inputs and outputs are not of equal length [number of samples]")

        return grid_paths,init_paths,opt_paths,lr_bnd_paths,odo_paths

    @staticmethod
    def __load_npz(file_path:str,array_name:str,data_type=np.float32, normalize_factor:float=1.0):

        npz_array =  np.load(file_path)[array_name]
        npz_array = (npz_array/normalize_factor).astype(data_type)
        return npz_array
    
    @staticmethod
    def __get_names(path:str)->Tuple[str,str,str,str]:
        list_split_path=path.split(os.sep)
        scene_name = list_split_path[-3]
        folder_name = list_split_path[-2]
        file_name = list_split_path[-1]
        file_prefix = re.split("_nos",file_name)[0]

        return scene_name,folder_name,file_name,file_prefix

    @staticmethod
    def __assert_lengths(in1,in2,in3,in4,in5,in6,in7):
            try:
                assert (
                len(in1)==len(in2)==len(in3)==len(in3)==len(in4)==len(in5)==len(in6)==len(in7)
                )
            except AssertionError:
                print(f"Assertion error, all inputs and outputs are not of equal length")

    @staticmethod
    def __create_example(grid_map, grid_org_res, left_bnd, right_bnd, car_odo, init_path, opt_path):

        def gridmap_feature(value):
            """Returns a bytes_list from a string / byte."""
            return tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[value.tostring()])
            )

        def bytes_feature(value):
            """Returns a bytes_list from a string / byte."""
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tostring()]))
            
        feature = {
            # model inputs
            "grid_map": gridmap_feature(grid_map),
            "grid_org_res":bytes_feature(grid_org_res),
            "left_bnd": bytes_feature(left_bnd),
            "right_bnd": bytes_feature(right_bnd),
            "car_odo": bytes_feature(car_odo),
            "init_path": bytes_feature(init_path),
            # model outputs
            "opt_path": bytes_feature(opt_path),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    
    def __write_records(self,grid_path,**file_data):


        grid_map = file_data.get("in_grid_map")
        grid_org_res = file_data.get("in_grid_org_res")
        left_bnd = file_data.get("in_left_bnd")
        right_bnd = file_data.get("in_right_bnd")
        car_odo = file_data.get("in_car_odo")
        init_path = file_data.get("in_init_path")

        opt_path = file_data.get("out_opt_path")

        scene_name,folder_name,file_name,file_prefix = self.__get_names(grid_path)

        current_tfrec_dir = os.path.join(self.target_path,scene_name,folder_name)

        num_tfrecods = len(grid_map) // self.num_samples


        if len(grid_map) % self.num_samples:
            num_tfrecods += 1  # add one record if there are any remaining samples
        
        logger.info(f"file:{file_name} having {len(grid_map)} samples, creating {num_tfrecods} records")

        if not os.path.exists(current_tfrec_dir):
            os.makedirs(current_tfrec_dir)

        for tfrec_num in range(num_tfrecods):
            samples = grid_map[(tfrec_num * self.num_samples) : ((tfrec_num + 1) * self.num_samples)]

            with tf.io.TFRecordWriter(
                f"{current_tfrec_dir}/{file_prefix}_file_{tfrec_num:02d}-{len(samples)}.tfrec"
            ) as writer:
                for s in range(len(samples)):
                    #print(np.shape(np_init_path[i]))
                    example = self.__create_example(
                        grid_map[s],grid_org_res[s],
                        left_bnd[s],right_bnd[s],
                        car_odo[s],init_path[s],
                        opt_path[s])
                    writer.write(example.SerializeToString())

    def create_records(self):

        all_grid_paths,all_init_paths,all_opt_paths,all_lr_bnd_paths,all_odo_paths = self.__get_all_msg_paths()

        #logger.info(len(all_grid_paths),len(all_init_paths),len(all_opt_paths),len(all_lr_bnd_paths),len(all_odo_paths))
        
        for i in trange(len(all_grid_paths)):
            # load inputs    
            in_grid_map = self.__load_npz(file_path=all_grid_paths[i],
                array_name="grid_data", data_type=np.int8,normalize_factor=127.0)
            in_grid_org_res = self.__load_npz(all_grid_paths[i],"grid_org_res")
            in_init_path = self.__load_npz(all_init_paths[i],"init_path")
            in_left_bnd = self.__load_npz(all_lr_bnd_paths[i],"left_bnd")
            in_right_bnd = self.__load_npz(all_lr_bnd_paths[i],"right_bnd")
            in_car_odo = self.__load_npz(all_odo_paths[i],"odo_data")
            
            # load outputs
            out_opt_path = self.__load_npz(all_opt_paths[i],"opt_path")

            self.__assert_lengths(in_grid_map,in_grid_org_res,
            in_init_path,out_opt_path,in_left_bnd,in_right_bnd,in_car_odo)

            """
            logger.info(f"\n \
            Grid map:{np.shape(in_grid_map)}\n \
            Grid Orig resol:{np.shape(in_grid_org_res)}\n \
            Init path:{np.shape(in_init_path)}\n \
            Opt Path:{np.shape(out_opt_path)}\n \
            Left bnd:{np.shape(in_left_bnd)}\n \
            Right bnd{np.shape(in_right_bnd)}\n \
            Car x,y,theta:{np.shape(in_car_odo)}\n")
            """
            dict_data = {
                "in_grid_map":in_grid_map,
                "in_grid_org_res":in_grid_org_res,
                "in_left_bnd":in_left_bnd,
                "in_right_bnd":in_right_bnd,
                "in_car_odo":in_car_odo,
                "in_init_path":in_init_path,
                'out_opt_path':out_opt_path

            }

            # write tf records
            self.__write_records(all_grid_paths[i],**dict_data)


s_path = r'D:\npz_files'
tgt_path = r'D:\tf_records'

tf_rec_maker = Tfrecs_frm_npz(
    source_path=s_path,
    target_path=tgt_path,
    samples_per_record=16)

tf_rec_maker.create_records()

