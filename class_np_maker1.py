import rosbag
#from cv_bridge import CvBridge, CvBridgeError
import PIL.Image as Image 
import numpy as np
from numpy import save
from numpy import load
from tqdm import tqdm
import sys
from typing import List,Any
import io
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

    def create_np_path(self):

        list_all_init_path = []
        list_all_opt_path = []
        print("Converting path data to numpy........ ")
        for i,examplary_pathing_msg in enumerate(tqdm(self.pathing_msgs)):

            #print("examplary_pathing_msg")

            init_path_x,init_path_y = self.__construct_path(examplary_pathing_msg.path_options[0].reference_path)

            opt_path_x,opt_path_y = self.__construct_path(examplary_pathing_msg.path_options[2].reference_path)

            if _ZERO_PADDING:
                if len(init_path_x)<self.max_length:
                    init_path_x.extend(self.__padd_values(self.max_length-len(init_path_x),0))
                    #init_path_x = self.__padded_path(self,init_path_x)
                    init_path_y.extend(self.__padd_values(self.max_length-len(init_path_y),0))
            
                if len(opt_path_x)<self.max_length:
                    opt_path_x.extend(self.__padd_values(self.max_length-len(opt_path_x),0))
                    opt_path_y.extend(self.__padd_values(self.max_length-len(opt_path_y),0))
                        
            else:                    
                if len(init_path_x)<self.max_length:
                    init_path_x.extend(self.__padd_values(self.max_length-len(init_path_x),init_path_x[-1]))
                    #init_path_x = self.__padded_path(self,init_path_x)
                    init_path_y.extend(self.__padd_values(self.max_length-len(init_path_y),init_path_y[-1]))
            
                if len(opt_path_x)<self.max_length:
                    opt_path_x.extend(self.__padd_values(self.max_length-len(opt_path_x),opt_path_x[-1]))
                    opt_path_y.extend(self.__padd_values(self.max_length-len(opt_path_y),opt_path_y[-1]))

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

    def create_np_grid(self):

        list_all_grid_data =[]
        print("Converting grid data to numpy........ ")
        for i,examplary_pathing_msg in enumerate(tqdm(self.pathing_msgs)):

            #get sequences aligning with pathing messages
            grid_seq = examplary_pathing_msg.path_options[1].reference_path[0].left_boundaries.lane

            """
            grid is saved as vector in grid_msg.data
            grid_msg.info gives information about properties, i.e. width, height, resolution, position in odometry frame
            """
            grid_idx = grid_seqs.index(grid_seq)
            grid_msg = grid_msgs[grid_idx]

            grid_data = np.asarray(grid_msg.data)
            grid_data = grid_data.reshape(grid_msg.info.width, grid_msg.info.height)
            
            # append grid data
            list_all_grid_data.append(grid_data) 

        #converted appended grids to np array
        np_all_grid_data = np.asarray(list_all_grid_data)

        return np_all_grid_data

    def create_np_img(self):
        list_all_img_data =[]
        print("Converting image data to numpy........ ")
        for i,examplary_pathing_msg in enumerate(tqdm(self.pathing_msgs)):

            image_seq = examplary_pathing_msg.path_options[1].reference_path[0].left_boundaries.road
            image_idx = image_seqs.index(image_seq)
            image_msg = image_msgs[image_idx]

            # Convert compressed image(byte string) to RAW using
            img = Image.open(io.BytesIO(image_msg.data))
            #convert PIL image object to numpy array
            np_img = np.asarray(img)
            list_all_img_data.append(np_img)

        np_all_img_data = np.asarray(list_all_img_data)

        return np_all_img_data

    def create_np_odo(self):
        list_all_odo_data = []
        print("Converting Odometer data to numpy........ ")
        for examplary_pathing_msg in enumerate(tqdm(self.pathing_msgs)):

            odom_seq = examplary_pathing_msg.path_options[1].reference_path[0].left_boundaries.obstacle
            odom_idx = odom_seqs.index(odom_seq)
            odom_msg = odom_msgs[odom_idx]

            #adding only position (x,y), check odo_msg for other data(heading, angle and others) thats available
            list_all_odo_data.append((odom_msg.pos_x,odom_msg.pos_y))
        
        np_all_odo_data = np.asfarray(list_all_odo_data)

        return np_all_odo_data

# functionality check
convert_2_np = np_maker(pathing_msgs,_MAX_LENGTH)

np_init_path,np_opt_path = convert_2_np.create_np_path()

np_grid_data = convert_2_np.create_np_grid()

np_image_data = convert_2_np.create_np_img()


print((np_init_path[0]))
print(np.shape(np_opt_path))

print(np.shape(np_grid_data))

print(np.shape(np_image_data))

from matplotlib import pyplot as plt
#plt.matshow(np_grid_data[0])

plt.plot(np_init_path[0,:15,0],np_init_path[0,:15,1])
plt.show()

