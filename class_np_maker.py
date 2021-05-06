import rosbag
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from numpy import save
from numpy import load
from tqdm import tqdm
import sys
from typing import List,Any

sys.path.insert(0, '~/ddad/application/urban/remote_services/teleop_machine_learning/src/training')

_READ_ROSBAG =1
_MAX_LENGTH = 25
_ZERO_PADDING = 1

bag = rosbag.Bag('2021-03-12_11-41-55_0.bag')
topic_list = ['/em/fused_grid', '/sensorik/axis_front/image/compressed', '/vehicle/odometry', '/function/pathing_mlteleop']

if _READ_ROSBAG:
    pathing_msgs = []
    grid_msgs = []
    grid_seqs = []
    odom_msgs = []
    odom_seqs = []
    image_msgs = []
    image_seqs = []
    for topic, msg, t in bag.read_messages(topics=topic_list):
        if topic == "/function/pathing_mlteleop":
            pathing_msgs.append(msg)

        if topic == "/em/fused_grid":
            grid_msgs.append(msg)
            grid_seqs.append(msg.header.seq)
        if topic == "/vehicle/odometry":
            odom_msgs.append(msg)
            odom_seqs.append(msg.header.seq)
        if topic == "/sensorik/axis_front/image/compressed":
            image_msgs.append(msg)
            image_seqs.append(msg.header.seq)

    bag.close()


class np_maker():

    def __init__(self,pathing_msgs,max_length):
        self.pathing_msgs = pathing_msgs
        self.max_length = max_length
        self.extract_msg_cnt = 220

    @staticmethod
    def __np_reshape_frm_list(list_path,new_shape):
        return np.reshape((np.asarray(list_path)),new_shape)
    
    @staticmethod
    def __np_hstack_list(self,list1,list2,new_shape):
        assert len(list1)==len(list2)
        np_list1 = self.__np_reshape_frm_list(list1,new_shape)
        np_list2 = self.__np_reshape_frm_list(list2,new_shape)
        return np.hstack((np_list1,np_list2))
    
    @staticmethod
    def __padd_values(n,val):
        listofzeros = [val] * n
        return listofzeros
    
    @staticmethod
    def __padded_path(self,unpadded_path):
        padded_path = unpadded_path.extend(self.__padd_values(self.max_length-len(unpadded_path),0))
        return padded_path

    @staticmethod
    def __construct_path(list_of_points):
        path_x = [] 
        path_y = []
        for path in list_of_points:
            path_x.append(path.point.x)
            path_y.append(path.point.y)

        return path_x,path_y  

    @staticmethod
    def __assert_all_lengths(self,path1_x,path1_y,path2_x,path2_y):
        # check if every path is as long as the maximum length
        assert len(path1_x)==self.max_length 
        assert len(path1_y)==self.max_length 
        assert len(path2_x)==self.max_length 
        assert len(path2_y)==self.max_length

    def create_np_array(self):

        list_all_init_path = []
        list_all_opt_path = []

        for i,examplary_pathing_msg in enumerate(tqdm(self.pathing_msgs[0:3])):

            print("examplary_pathing_msg")

            '''
            for init_path in examplary_pathing_msg.path_options[0].reference_path:
                #tup_init_xy.append((init_path.point.x,init_path.point.y))
                init_path_x.append(init_path.point.x)
                init_path_y.append(init_path.point.y)
            '''
            init_path_x,init_path_y = self.__construct_path(examplary_pathing_msg.path_options[0].reference_path)

            '''
            for opt_path in examplary_pathing_msg.path_options[2].reference_path:
                opt_path_x.append(opt_path.point.x)
                opt_path_y.append(opt_path.point.y)
            '''
            opt_path_x,opt_path_y = self.__construct_path(examplary_pathing_msg.path_options[2].reference_path)

            if _ZERO_PADDING:
                if len(init_path_x)<self.max_length:
                    init_path_x.extend(self.__padd_values(self.max_length-len(init_path_x),0))
                    #init_path_x = self.__padded_path(self,init_path_x)
                    init_path_y.extend(self.__padd_values(self.max_length-len(init_path_y),0))
            
                if len(opt_path_x)<self.max_length:
                    opt_path_x.extend(self.__padd_values(self.max_length-len(opt_path_x),0))
                    opt_path_y.extend(self.__padd_values(self.max_length-len(opt_path_y),0))
            
            #check for length
            self.__assert_all_lengths(self,init_path_x,init_path_y,opt_path_x,opt_path_y)

            #pair x and y and 
            np_init_path = self.__np_hstack_list(self,init_path_x,init_path_y,new_shape=(self.max_length,1))
            np_opt_path = self.__np_hstack_list(self,opt_path_x,opt_path_y,new_shape=(self.max_length,1))
            
            # append paths from all pathing messages
            list_all_init_path.append(np_init_path)
            list_all_opt_path.append(np_opt_path)

        # Convert the all paths appended list to numpy array
        np_all_initp = np.asarray(list_all_init_path)
        np_all_optp = np.asarray(list_all_opt_path)        

        return np_all_initp,np_all_optp

convert_2_np = np_maker(pathing_msgs,_MAX_LENGTH)

np_init_path,np_opt_path = convert_2_np.create_np_array()

print((np_init_path[0]))
print(np.shape(np_opt_path))

